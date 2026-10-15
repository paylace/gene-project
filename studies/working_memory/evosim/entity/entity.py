"""개체(Entity) 본체: 유전체 초기화/복제/무손실 스냅샷 + 에피소드 평가(학습/동결 분리). 세부 기능은 Mixin으로 분리됨."""

from collections import defaultdict, deque

from ..config import CONFIG
from ..module import Module
from .learning import LearningMixin
from .mutation import MutationMixin
from .network import NetworkMixin


# --- 개체 (순방향 DAG + 시간 지연 순환 간선 + 출력층 가소성 & TD 학습) ---
class Entity(NetworkMixin, LearningMixin, MutationMixin):
    def __init__(
        self,
        genome=None,
        value_estimates=None,
        allow_recurrent=None,
        decay_values=True,
        allow_recurrent_edges=None,
        allow_module_memory=None,
    ):
        self.allow_recurrent = (
            CONFIG["ALLOW_RECURRENT"] if allow_recurrent is None else allow_recurrent
        )
        cfg_rec_edges = CONFIG.get("ALLOW_RECURRENT_EDGES")
        cfg_mod_mem = CONFIG.get("ALLOW_MODULE_MEMORY")

        self.allow_recurrent_edges = (
            (self.allow_recurrent if cfg_rec_edges is None else cfg_rec_edges)
            if allow_recurrent_edges is None
            else allow_recurrent_edges
        )
        self.allow_module_memory = (
            (self.allow_recurrent if cfg_mod_mem is None else cfg_mod_mem)
            if allow_module_memory is None
            else allow_module_memory
        )

        if genome is None:
            self.modules = {
                f"m{i}": Module() for i in range(CONFIG["INITIAL_MODULES"])
            }
            self.connections = defaultdict(dict)            # target -> {source: weight} (t -> t)
            self.recurrent_connections = defaultdict(dict)  # target -> {source: weight} (t-1 -> t)
            self._generate_initial_topology()
        else:
            self.modules = {
                k: Module(**v) for k, v in genome["modules"].items()
            }
            self.connections = defaultdict(
                dict, {k: dict(v) for k, v in genome["connections"].items()}
            )
            self.recurrent_connections = defaultdict(
                dict, {k: dict(v) for k, v in genome.get("recurrent_connections", {}).items()}
            )

        # 다중 스텝 TD Q-테이블: (state_key, action) -> Q-value
        # - 자손 번식(decay_values=True): LAMARCKIAN_DECAY(0.5) 적용
        # - 평가용 무손실 스냅샷(decay_values=False): Q-value를 감쇠 없이 그대로 보존
        self.value_estimates = defaultdict(float)
        if value_estimates:
            decay = CONFIG["LAMARCKIAN_DECAY"] if decay_values else 1.0
            for k, v in value_estimates.items():
                self.value_estimates[k] = v * decay

        self.experience_buffer = deque(maxlen=CONFIG["EXPERIENCE_BUFFER_SIZE"])
        self.prev_outputs = {}   # t-1 시점의 각 노드 활성화 상태 (작업 기억 버퍼)
        self.last_traces = {}    # 최근 스텝의 활성 흔적(Eligibility Trace)
        self.fitness = 0.0
        self.accuracy = 0.0
        self.cue_accuracy = 0.0
        self.memory_accuracy = 0.0

        # 훈련 지표 / 검증 지표 / 최종 테스트 지표 명시적 분리 저장
        self.training_metrics = None
        self.validation_metrics = None
        self.test_metrics = None

    def complexity_cost(self):
        edge_count = sum(len(v) for v in self.connections.values()) + sum(
            len(v) for v in self.recurrent_connections.values()
        )
        return CONFIG["COMPLEXITY_PENALTY"] * (len(self.modules) + 0.5 * edge_count)

    def evaluate(
        self,
        env,
        num_episodes=None,
        learn=True,
        trajectories=None,
        rng=None,
        update_metrics=None,
        return_detailed=False,
    ):
        """
        다중 스텝 에피소드 평가 (단서 구간 정확도와 작업 기억 지연 구간 정확도를 분리 측정).

        Args:
            env: Environment 인스턴스
            num_episodes: 평가 에피소드 수 (trajectories가 주어지면 len(trajectories) 사용)
            learn: True이면 에피소드 종료 후 TD/Hebbian 업데이트 및 경험 재생 수행.
                   False(동결 평가)이면 가중치, Q-테이블, 경험 버퍼를 일절 수정하지 않음.
            trajectories: 사전 생성된 에피소드 리스트. 주어지면 환경에서 새로 샘플링하지 않음.
            rng: 행동 선택 및 에피소드 생성에 사용할 독립 난수 생성기(random.Random).
                 주어지면 전역 훈련 난수 스트림을 전혀 소비하지 않음.
            update_metrics: True이면 개체의 self.fitness/accuracy 속성을 갱신함.
                            기본값(None)일 때는 learn과 동일하게 동작하여, 동결 검증이 훈련 지표를
                            덮어쓰지 않도록 보호함.
            return_detailed: True이면 에피소드 단위 전 스텝 정답률 및 지연 시점별 정확도를 포함한
                             상세 dict를 5번째 반환값으로 추가하거나 evaluate_frozen에서 활용함.
        """
        if update_metrics is None:
            update_metrics = learn

        if trajectories is not None:
            episode_list = trajectories
            effective_episodes = len(episode_list)
        else:
            effective_episodes = (
                CONFIG["EPISODES_PER_EVAL"] if num_episodes is None else num_episodes
            )
            episode_list = None

        total_reward = 0.0
        total_steps = 0
        correct_total = 0
        cue_steps = 0
        cue_correct = 0
        masked_steps = 0
        masked_correct = 0

        # 에피소드 단위 및 시점별(t) 세분화 통계
        episodes_with_mask = 0
        episodes_all_masked_correct = 0
        step_total_counts = defaultdict(int)
        step_correct_counts = defaultdict(int)

        for ep in range(effective_episodes):
            if episode_list is not None:
                trajectory = episode_list[ep]
            else:
                target_class = ep % 3
                trajectory = env.generate_episode_trajectory(
                    target_class=target_class, rng=rng
                )

            self.reset_memory()

            transition_records = []
            ep_masked_total = 0
            ep_masked_correct = 0

            for step_idx, step_data in enumerate(trajectory):
                obs = step_data["obs"]
                correct_action = step_data["correct_action"]
                is_masked = step_data["is_masked"]
                t_step = step_data.get("step", step_idx)

                state_key = self._get_state_key(obs)
                logits, _ = self.forward(obs)
                action, probs = self.select_action(
                    state_key, logits, epsilon=0.0, rng=rng
                )

                reward = env.get_reward(action, probs, correct_action)
                total_reward += reward
                total_steps += 1
                step_total_counts[t_step] += 1

                is_correct = action == correct_action
                if is_correct:
                    correct_total += 1
                    step_correct_counts[t_step] += 1

                if is_masked:
                    masked_steps += 1
                    ep_masked_total += 1
                    if is_correct:
                        masked_correct += 1
                        ep_masked_correct += 1
                else:
                    cue_steps += 1
                    if is_correct:
                        cue_correct += 1

                if learn:
                    transition_records.append(
                        (state_key, action, reward, self.last_traces)
                    )

            if ep_masked_total > 0:
                episodes_with_mask += 1
                if ep_masked_correct == ep_masked_total:
                    episodes_all_masked_correct += 1

            if learn:
                # 다중 스텝 상태 전이(s_t -> s_{t+1})에 대한 TD 업데이트 및 경험 저장
                for t in range(len(transition_records)):
                    s_key, act, rew, step_traces = transition_records[t]
                    done = t == (len(transition_records) - 1)
                    next_s_key = None if done else transition_records[t + 1][0]
                    self.td_and_hebbian_update(
                        s_key, act, rew, next_s_key, done, traces=step_traces
                    )
                    self.store_experience(s_key, act, rew, next_s_key, done)

        if learn:
            # Preserve the original training RNG path, but isolate explicitly
            # seeded paired adaptation experiments including replay sampling.
            if rng is None:
                self.replay_experience()
            else:
                self.replay_experience(rng=rng)

        comp_cost = self.complexity_cost()
        fitness = (total_reward / max(1, effective_episodes)) - comp_cost
        accuracy = correct_total / max(1, total_steps)
        cue_accuracy = cue_correct / max(1, cue_steps)
        memory_accuracy = masked_correct / max(1, masked_steps)
        episode_full_memory_acc = episodes_all_masked_correct / max(1, episodes_with_mask)
        step_accuracies = {
            t: step_correct_counts[t] / max(1, step_total_counts[t])
            for t in sorted(step_total_counts.keys())
        }

        metrics_dict = {
            "fitness": fitness,
            "accuracy": accuracy,
            "cue_accuracy": cue_accuracy,
            "memory_accuracy": memory_accuracy,
            "episode_full_memory_accuracy": episode_full_memory_acc,
            "step_accuracies": step_accuracies,
        }

        if update_metrics:
            self.fitness = fitness
            self.accuracy = accuracy
            self.cue_accuracy = cue_accuracy
            self.memory_accuracy = memory_accuracy
            if learn:
                self.training_metrics = metrics_dict

        if return_detailed:
            return fitness, accuracy, cue_accuracy, memory_accuracy, metrics_dict
        return fitness, accuracy, cue_accuracy, memory_accuracy

    def evaluate_frozen(self, env, trajectories=None, num_episodes=None, rng=None):
        """
        무손실 평가 스냅샷 위에서 가중치와 Q값을 동결한 채 성능을 측정하고 상세 지표 사전을 반환.
        원본 개체의 훈련 상태, 가중치, Q-테이블, 작업 기억, 지표를 일절 변경하지 않음.
        """
        snap = self.snapshot_for_eval()
        _, _, _, _, details = snap.evaluate(
            env,
            num_episodes=num_episodes,
            learn=False,
            trajectories=trajectories,
            rng=rng,
            update_metrics=False,
            return_detailed=True,
        )
        return details

    def copy(self, preserve_values=True, decay_values=True):
        """
        개체 복제.
        - 세대 간 자손/엘리트 번식: preserve_values=True, decay_values=True (기본값, LAMARCKIAN_DECAY 적용)
        - 평가용 무손실 스냅샷: preserve_values=True, decay_values=False (Q값 100% 보존)
        """
        genome = {
            "modules": {k: v.to_dict() for k, v in self.modules.items()},
            "connections": {k: dict(v) for k, v in self.connections.items()},
            "recurrent_connections": {
                k: dict(v) for k, v in self.recurrent_connections.items()
            },
        }
        val_est = self.value_estimates if preserve_values else None
        return Entity(
            genome,
            value_estimates=val_est,
            allow_recurrent=self.allow_recurrent,
            decay_values=decay_values,
            allow_recurrent_edges=self.allow_recurrent_edges,
            allow_module_memory=self.allow_module_memory,
        )

    def snapshot_for_eval(self):
        """
        평가 스냅샷 계약(Evaluation Snapshot Contract):
        - 모듈 연산자/바이어스/감쇠율, 순방향/순환 연결 가중치, Q-테이블(감쇠 없는 원본 값),
          순환/기억 허용 설정을 무손실 복제함.
        - 작업 기억(prev_outputs)과 경험 버퍼(experience_buffer)는 빈 상태로 초기화하여
          각 동결 평가 에피소드가 독립적인 초기 작업 기억에서 시작하도록 보장함.
        """
        snap = self.copy(preserve_values=True, decay_values=False)
        if self.training_metrics is not None:
            snap.training_metrics = dict(self.training_metrics)
        if self.validation_metrics is not None:
            snap.validation_metrics = dict(self.validation_metrics)
        if self.test_metrics is not None:
            snap.test_metrics = dict(self.test_metrics)
        return snap
