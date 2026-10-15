"""Entity의 생애 주기 학습 담당: 상태 이산화, 행동 선택, TD Q-Learning, 출력층 보상 조절 가소성, 경험 재생"""

import math
import random

from ..config import CONFIG


class LearningMixin:
    """다중 스텝 TD Q-테이블 + 보상 조절 출력층 가중치 가소성"""

    def _get_state_key(self, stimuli):
        """
        Q-테이블용 POMDP 상태 키 생성.

        - 기본(Q_STATE_INCLUDES_MEMORY=False): 관측값만 이산화. 실험군/대조군 모두 동일한 키를 사용하므로
          수작업 상태 키 기억 채널을 제거함.
        - True(레거시): 순환 허용 개체에 한해 t-1 모듈 활성값과 직전 출력 argmax(`prev_out_max`)를 키에 추가.
        """
        if stimuli is None:
            return None
        bins = CONFIG["STATE_BINS"]
        obs_key = tuple(round(max(-0.5, min(1.5, s)) * bins) / bins for s in stimuli)

        if CONFIG["Q_STATE_INCLUDES_MEMORY"] and self.allow_recurrent and self.modules:
            mem_vals = [
                self.prev_outputs.get(m, 0.0) for m in sorted(self.modules.keys())[:2]
            ]
            prev_out_max = (
                max(range(3), key=lambda i: self.prev_outputs.get(f"out_{i}", 0.0))
                if any(abs(self.prev_outputs.get(f"out_{i}", 0.0)) > 1e-4 for i in range(3))
                else -1
            )
            mem_key = tuple(round(max(-1.0, min(1.0, v)) * 3) / 3 for v in mem_vals) + (
                prev_out_max,
            )
        else:
            mem_key = ()
        return obs_key + mem_key

    @staticmethod
    def _softmax(logits):
        max_l = max(logits)
        exps = [math.exp(max(-10.0, min(10.0, l - max_l))) for l in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def select_action(self, state_key, logits, epsilon=0.0, rng=None):
        """
        신경망 소프트맥스 확률과 다중 스텝 TD Q-테이블을 결합한 하이브리드 행동 선택.
        - rng 인자가 주어지면 전역 random 스트림을 소비하지 않고 독립 난수원을 사용함.
        - value_estimates 조회 시 .get()을 사용하여 동결 평가(learn=False) 중 defaultdict에
          새 키가 삽입되는 부작용을 차단함.
        """
        r = random if rng is None else rng
        probs = self._softmax(logits)
        if r.random() < epsilon:
            return r.randint(0, 2), probs

        td_weight = CONFIG["TD_WEIGHT"]
        combined_scores = []
        for action in range(3):
            q_val = self.value_estimates.get((state_key, action), 0.0)
            score = (1.0 - td_weight) * probs[action] + td_weight * math.tanh(q_val)
            combined_scores.append(score)

        max_score = max(combined_scores)
        best_actions = [
            i for i, s in enumerate(combined_scores) if math.isclose(s, max_score, abs_tol=1e-9)
        ]
        return r.choice(best_actions), probs

    def td_and_hebbian_update(
        self, state_key, action, reward, next_state_key, done, traces=None
    ):
        """
        1. 다중 스텝 TD(0) Q-Learning 업데이트 (next_state_key 반영)
        2. 보상 조절 출력층 가중치 가소성: TD 오차(td_error)를 조절 신호로 삼아
           선택된 행동의 출력 노드(`out_{action}`)로 들어오는 활성 간선 가중치를 미세 조정.

        Args:
            traces: 해당 전이(state_key, action)가 발생한 스텝의 활성 흔적 (`forward()` 직후의
                `self.last_traces`). None이면 `self.last_traces`(가장 최근 스텝)를 사용.
        """
        lr = CONFIG["TD_LEARNING_RATE"]
        gamma = CONFIG["TD_DISCOUNT"]

        current_q = self.value_estimates.get((state_key, action), 0.0)
        if done or next_state_key is None:
            next_max_q = 0.0
        else:
            next_max_q = max(
                self.value_estimates.get((next_state_key, a), 0.0) for a in range(3)
            )

        td_target = reward + gamma * next_max_q
        td_error = td_target - current_q
        self.value_estimates[(state_key, action)] = current_q + lr * td_error

        # 보상 조절 출력층 가소성 (선택된 출력 노드로 유입되는 순방향/순환 간선 가중치 조정)
        hebb_lr = CONFIG["HEBBIAN_LR"]
        clipped_delta = max(-1.0, min(1.0, td_error))

        if traces is None:
            traces = self.last_traces

        target_out = f"out_{action}"
        if hebb_lr != 0.0 and target_out in traces:
            out_trace, _ = traces[target_out]
            for (edge_type, src), pre_val in out_trace.items():
                delta_w = hebb_lr * clipped_delta * pre_val
                if edge_type == "ff" and src in self.connections[target_out]:
                    self.connections[target_out][src] = max(
                        -3.0, min(3.0, self.connections[target_out][src] + delta_w)
                    )
                elif edge_type == "rec" and src in self.recurrent_connections[target_out]:
                    self.recurrent_connections[target_out][src] = max(
                        -3.0, min(3.0, self.recurrent_connections[target_out][src] + delta_w)
                    )

        return td_error

    def store_experience(self, state_key, action, reward, next_state_key, done):
        self.experience_buffer.append((state_key, action, reward, next_state_key, done))

    def replay_experience(self, rng=None):
        if len(self.experience_buffer) < 12:
            return
        r = random if rng is None else rng
        batch = r.sample(self.experience_buffer, min(16, len(self.experience_buffer)))
        lr = CONFIG["TD_LEARNING_RATE"]
        gamma = CONFIG["TD_DISCOUNT"]
        for state_key, action, reward, next_state_key, done in batch:
            current_q = self.value_estimates.get((state_key, action), 0.0)
            next_max_q = (
                0.0
                if (done or next_state_key is None)
                else max(self.value_estimates.get((next_state_key, a), 0.0) for a in range(3))
            )
            self.value_estimates[(state_key, action)] = current_q + lr * (
                reward + gamma * next_max_q - current_q
            )
