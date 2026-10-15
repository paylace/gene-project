"""v0.1 순서의 수치 계산과 조건 검증. 관찰기·전체 평가기는 W3에서 추가한다."""

import math
from .network import INPUTS, OUTPUTS, node_key


class NetworkPolicy:
    def __init__(self, genome, settings):
        genome.validate(settings)
        self.settings = settings
        self.genome = genome.clone()
        self.nodes = list(INPUTS) + genome.topological_order() + list(OUTPUTS)
        self.index = {node: i for i, node in enumerate(self.nodes)}
        self.plan = []
        for node in self.nodes[4:]:
            ff = [(self.index[src], weight) for src, weight in sorted(genome.ff.get(node, {}).items(), key=lambda p: node_key(p[0]))]
            rec = [(self.index[src], weight) for src, weight in sorted(genome.rec.get(node, {}).items(), key=lambda p: node_key(p[0]))]
            self.plan.append((self.index[node], genome.modules.get(node), ff, rec))
        self.reset()

    def reset(self):
        self.previous = [0.0] * len(self.nodes)

    def forward(self, observation):
        if len(observation) != 4 or not all(math.isfinite(v) for v in observation):
            raise ValueError("Expected four finite observations")
        current = list(observation) + [0.0] * (len(self.nodes) - 4)
        for idx, module, ff, rec in self.plan:
            weighted = 0.0
            for src, weight in ff:
                weighted += current[src] * weight
            for src, weight in rec:
                weighted += self.previous[src] * weight
            current[idx] = module.compute(weighted, self.previous[idx]) if module else weighted
        if not all(math.isfinite(v) for v in current):
            raise FloatingPointError("Nonfinite network activation")
        self.previous = current
        return current[-3:]

    def act(self, observation, rng):
        logits = self.forward(observation)
        maximum = max(logits)
        exps = [math.exp(max(-10.0, min(10.0, logit - maximum))) for logit in logits]
        total = sum(exps)
        probs = [value / total for value in exps]
        best = max(probs)
        choices = [i for i, p in enumerate(probs) if math.isclose(p, best, rel_tol=self.settings.parameters["tie_rel_tol"], abs_tol=self.settings.parameters["tie_abs_tol"])]
        return choices[0] if len(choices) == 1 else rng.choice(choices)


class BatchNetwork:
    """Independent episodes in NumPy columns; same ordered graph operations."""

    def __init__(self, genome, count, settings):
        import numpy as np
        self.policy = NetworkPolicy(genome, settings)
        self.previous = np.zeros((len(self.policy.nodes), count), dtype=float)

    def forward(self, observations):
        import numpy as np
        current = np.zeros_like(self.previous)
        current[:4] = observations.T
        for idx, module, ff, rec in self.policy.plan:
            weighted = np.zeros(current.shape[1])
            for src, weight in ff:
                weighted += current[src] * weight
            for src, weight in rec:
                weighted += self.previous[src] * weight
            if module is None:
                current[idx] = weighted
                continue
            drive = weighted + module.bias
            clipped = np.clip(drive, -10.0, 10.0)
            if module.op == "linear":
                value = 1 / (1 + np.exp(-clipped))
            elif module.op == "threshold":
                value = (drive > 0).astype(float)
            elif module.op == "relu":
                value = np.clip(drive, 0.0, 2.0)
            elif module.op == "tanh":
                value = np.tanh(clipped)
            elif module.op == "ema":
                value = (1 - module.memory_decay) * np.tanh(clipped) + module.memory_decay * self.previous[idx]
            else:
                value = np.where(np.abs(weighted) > 0.35, np.tanh(clipped), self.previous[idx] * module.memory_decay)
            current[idx] = value
        if not np.isfinite(current).all():
            raise FloatingPointError("Nonfinite network activation")
        self.previous = current
        return current[-3:].T
