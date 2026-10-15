import random

from .config import CONFIG


# --- 다중 스텝 부분 관측 환경 (Multi-Step POMDP for Working Memory) ---
class Environment:
    """
    작업 기억(Working Memory) 창발을 검증하기 위한 다중 스텝 POMDP 환경:
    - 에피소드는 총 T(STEPS_PER_EPISODE=4) 스텝으로 진행됨.
    - 초기 스텝에서는 원본 자극 [s1, s2, s3]가 제공되지만, 후반 mask_steps 동안에는
      핵심 자극(s1, s2, s3)이 0.0으로 가려짐(Sensory Occlusion).
    - 에이전트가 마스킹 구간에서도 정답 행동을 유지하려면 순환 간선(Recurrent Edge)이나
      모듈 내부 잔류 상태(Leaky Integrator / Latch)를 통해 작업 기억을 형성해야 함.
    """

    def __init__(self, stage_config):
        self.noise = stage_config["noise"]
        self.mask_steps = stage_config.get("mask_steps", 1)
        self.threshold = stage_config.get("threshold", CONFIG["STAGE_SUCCESS_THRESHOLD"])
        self.steps_per_episode = CONFIG["STEPS_PER_EPISODE"]

    def generate_episode_trajectory(self, target_class=None, rng=None):
        """
        하나의 에피소드(T 스텝 시퀀스) 생성.
        rng가 주어지면 전역 random 스트림을 건드리지 않고 해당 rng 인스턴스만 사용함.
        """
        r = random if rng is None else rng

        if target_class is None:
            target_class = r.randint(0, 2)

        # 1. 기저 정답 상태(Clean Cue) 생성 (3개 클래스 1:1:1 균형)
        if target_class == 0:
            s1 = r.uniform(0.68, 1.0)
            s2 = r.uniform(0.0, 0.32)
            s3 = r.uniform(0.0, 1.0)
            correct_action = 0
        elif target_class == 1:
            s1 = r.uniform(0.0, 0.62)
            s2 = r.uniform(0.68, 1.0)
            s3 = r.uniform(0.0, 0.32)
            correct_action = 1
        else:
            while True:
                s1, s2, s3 = [r.random() for _ in range(3)]
                if not (s1 > 0.65 and s2 < 0.35) and not (s2 > 0.65 and s3 < 0.35):
                    break
            correct_action = 2

        base_cue = [s1, s2, s3]
        visible_steps = max(1, self.steps_per_episode - self.mask_steps)

        trajectory = []
        for step in range(self.steps_per_episode):
            is_masked = step >= visible_steps
            if not is_masked:
                # 단서 제시 구간 (관측 노이즈만 추가)
                obs = [
                    base_cue[0] + r.uniform(-self.noise, self.noise),
                    base_cue[1] + r.uniform(-self.noise, self.noise),
                    base_cue[2] + r.uniform(-self.noise, self.noise),
                ]
            else:
                # 지연/마스킹 구간: 외부 자극이 차단(0.0)되고 미세 배경 노이즈만 존재
                bg_noise = self.noise * 0.5
                obs = [
                    r.uniform(-bg_noise, bg_noise),
                    r.uniform(-bg_noise, bg_noise),
                    r.uniform(-bg_noise, bg_noise),
                ]

            trajectory.append(
                {
                    "step": step,
                    "obs": obs,
                    "is_masked": is_masked,
                    "correct_action": correct_action,
                }
            )

        return trajectory

    def generate_trajectories(self, num_episodes, seed=None, rng=None, shuffle_classes=True):
        """
        공통 검증(Validation) 및 최종 테스트(Test)를 위한 고정 에피소드 묶음 사전 생성.
        - 전역 훈련 난수 스트림과 완전히 분리된 독립 Random 인스턴스를 사용.
        - 클래스 비율(0, 1, 2)을 균등하게 맞춘 뒤, shuffle_classes=True이면 클래스 순서를 섞어
          고정된 0->1->2 반복 순서 편향을 방지함.
        """
        if rng is None:
            rng = random.Random(seed)
        classes = [ep % 3 for ep in range(num_episodes)]
        if shuffle_classes:
            rng.shuffle(classes)
        return [
            self.generate_episode_trajectory(target_class=c, rng=rng)
            for c in classes
        ]

    def get_reward(self, action, action_probs, correct_action):
        """
        범주형 대칭 보상 + 소프트맥스 마진 Shaping (한계점 5 해결):
        - 기존 abs(action - correct_action) 거리 페널티가 유발하던 중간 행동(Action 1) 편향을 제거.
        - 정답 클래스의 확률 마진을 보조 신호로 제공하여 미세한 가중치 진화 방향을 안내.
        """
        base_reward = (
            CONFIG["REWARD_CORRECT"] if action == correct_action else CONFIG["REWARD_WRONG"]
        )
        correct_prob = action_probs[correct_action]
        other_max_prob = max(action_probs[i] for i in range(3) if i != correct_action)
        margin_bonus = CONFIG["MARGIN_SHAPING_SCALE"] * (correct_prob - other_max_prob)
        return base_reward + margin_bonus
