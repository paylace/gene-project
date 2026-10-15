"""Four-input evolutionary network; parameters and temporal state are separate."""

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass

INPUTS = tuple(f"in_{i}" for i in range(4))
OUTPUTS = tuple(f"out_{i}" for i in range(3))
OPS = ("linear", "threshold", "relu", "tanh", "ema", "latch")
SETTINGS = {
    "input_count": 4, "output_count": 3, "initial_modules": 4, "max_modules": 9,
    "elite_ratio": 0.2, "crossover_rate": 0.65, "mutation_rate": 0.50,
    "tournament_size": 3, "complexity_penalty": 0.0,
    "recurrence_mode": "seeded_positive", "allow_recurrent_edges": True,
    "allow_module_memory": True, "lifetime_learning": False,
    "td_weight": 0.0, "hebbian_lr": 0.0,
    "mutation_weights": [0.42, 0.12, 0.06, 0.18, 0.22, 0.10],
    "module_ops": list(OPS), "module_bias_initial": [-0.4, 0.4],
    "module_decay_initial": [0.2, 0.9], "input_weight_initial": [-1.5, 1.5],
    "chain_weight_initial": [-1.2, 1.2], "initial_self_rec_probability": 0.7,
    "initial_self_rec_weight": [0.4, 1.1], "output_primary_weight": [0.6, 1.5],
    "output_aux_weight": [-1.0, 1.0], "output_module_rec_weight": [0.4, 1.2],
    "output_self_rec_weight": [0.5, 1.2], "weight_change_probability": 0.35,
    "weight_change_sigma": 0.35, "weight_bounds": [-3.0, 3.0],
    "module_op_change_probability": 0.25, "module_bias_change_probability": 0.35,
    "module_bias_change_sigma": 0.25, "module_bias_bounds": [-2.0, 2.0],
    "module_decay_change_probability": 0.35, "module_decay_change_sigma": 0.15,
    "module_decay_bounds": [0.0, 0.95], "new_module_self_rec_probability": 0.5,
    "new_module_self_rec_weight": [0.4, 1.0], "new_edge_weight": [-1.2, 1.2],
    "remove_edge_rec_probability": 0.4, "matching_gene_second_probability": 0.5,
    "second_parent_ff_probability": 0.3, "second_parent_rec_probability": 0.35,
    "tie_rel_tol": 1e-9, "tie_abs_tol": 1e-9,
}


def node_key(node):
    if node.startswith("in_"):
        return 0, int(node[3:])
    if node.startswith("m"):
        return 1, int(node[1:])
    if node.startswith("out_"):
        return 2, int(node[4:])
    raise ValueError(f"Invalid node: {node}")


@dataclass(frozen=True)
class Module:
    op: str
    bias: float
    memory_decay: float

    @classmethod
    def create(cls, rng):
        return cls(rng.choice(OPS), rng.uniform(-0.4, 0.4), rng.uniform(0.2, 0.9))

    def compute(self, weighted, previous):
        drive = weighted + self.bias
        clipped = max(-10.0, min(10.0, drive))
        if self.op == "linear":
            return 1.0 / (1.0 + math.exp(-clipped))
        if self.op == "threshold":
            return float(drive > 0)
        if self.op == "relu":
            return max(0.0, min(2.0, drive))
        if self.op == "tanh":
            return math.tanh(clipped)
        if self.op == "ema":
            return (1 - self.memory_decay) * math.tanh(clipped) + self.memory_decay * previous
        if self.op == "latch":
            return math.tanh(clipped) if abs(weighted) > 0.35 else previous * self.memory_decay
        raise ValueError(self.op)

    def mutate(self, rng):
        op, bias, decay = self.op, self.bias, self.memory_decay
        if rng.random() < 0.25:
            op = rng.choice(OPS)
        if rng.random() < 0.35:
            bias = max(-2.0, min(2.0, bias + rng.gauss(0, 0.25)))
        if rng.random() < 0.35:
            decay = max(0.0, min(0.95, decay + rng.gauss(0, 0.15)))
        return Module(op, bias, decay)


@dataclass
class ModuleIds:
    next_id: int = 4

    def allocate(self):
        result = f"m{self.next_id}"
        self.next_id += 1
        return result


@dataclass
class Genome:
    modules: dict[str, Module]
    ff: dict[str, dict[str, float]]
    rec: dict[str, dict[str, float]]

    def clone(self):
        return copy.deepcopy(self)

    def to_dict(self):
        return {"modules": {k: asdict(v) for k, v in self.modules.items()},
                "ff": copy.deepcopy(self.ff), "rec": copy.deepcopy(self.rec)}

    @classmethod
    def from_dict(cls, data):
        result = cls({k: Module(**v) for k, v in data["modules"].items()},
                     copy.deepcopy(data["ff"]), copy.deepcopy(data["rec"]))
        result.validate()
        return result

    @property
    def digest(self):
        data = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(data.encode()).hexdigest()

    def topological_order(self):
        remaining = set(self.modules)
        result = []
        while remaining:
            ready = sorted((node for node in remaining
                            if not (set(self.ff.get(node, {})) & remaining)), key=node_key)
            if not ready:
                raise ValueError("Feedforward cycle")
            result.extend(ready)
            remaining.difference_update(ready)
        return result

    def validate(self):
        if not 1 <= len(self.modules) <= SETTINGS["max_modules"]:
            raise ValueError("Invalid module count")
        for node, module in self.modules.items():
            if not node.startswith("m") or node_key(node)[0] != 1:
                raise ValueError("Invalid module id")
            if module.op not in OPS or not all(math.isfinite(x) for x in (module.bias, module.memory_decay)):
                raise ValueError("Invalid module")
            if not 0 <= module.memory_decay <= 0.95:
                raise ValueError("Invalid memory decay")
        targets = set(self.modules) | set(OUTPUTS)
        for mapping, sources in ((self.ff, set(self.modules) | set(INPUTS)),
                                 (self.rec, set(self.modules) | set(OUTPUTS))):
            for target, edges in mapping.items():
                if target not in targets or not set(edges) <= sources:
                    raise ValueError("Dangling or invalid edge")
                if not all(math.isfinite(weight) for weight in edges.values()):
                    raise ValueError("Nonfinite weight")
        self.topological_order()


def initial_genome(rng: random.Random) -> Genome:
    modules = {f"m{i}": Module.create(rng) for i in range(4)}
    ids = list(modules)
    ff, rec = {}, {}
    for i, node in enumerate(ids):
        ff[node] = {src: rng.uniform(-1.5, 1.5) for src in INPUTS}
        if i:
            ff[node][ids[i - 1]] = rng.uniform(-1.2, 1.2)
        if rng.random() < 0.7:
            rec[node] = {node: rng.uniform(0.4, 1.1)}
    for i, out in enumerate(OUTPUTS):
        ff[out] = {ids[i]: rng.uniform(0.6, 1.5), ids[(i + 1) % 4]: rng.uniform(-1.0, 1.0)}
        rec[out] = {ids[i]: rng.uniform(0.4, 1.2), out: rng.uniform(0.5, 1.2)}
    result = Genome(modules, ff, rec)
    result.validate()
    return result


def add_ff_if_acyclic(genome, source, target, weight):
    edges = genome.ff.setdefault(target, {})
    if source in edges:
        return False
    edges[source] = weight
    try:
        genome.topological_order()
    except ValueError:
        del edges[source]
        return False
    return True


def mutate(genome, rng, allocator, forced_kind=None):
    kinds = ("weight", "add_module", "remove_module", "add_ff", "add_rec", "remove_edge")
    kind = forced_kind or rng.choices(kinds, SETTINGS["mutation_weights"], k=1)[0]
    before = genome.digest
    added = None
    ids = sorted(genome.modules, key=node_key)
    if kind == "weight":
        for node in rng.sample(ids, min(2, len(ids))):
            genome.modules[node] = genome.modules[node].mutate(rng)
        for mapping in (genome.ff, genome.rec):
            for target in sorted(mapping, key=node_key):
                for src in sorted(mapping[target], key=node_key):
                    if rng.random() < 0.35:
                        mapping[target][src] = max(-3.0, min(3.0, mapping[target][src] + rng.gauss(0, 0.35)))
    elif kind == "add_module" and len(ids) < 9:
        added = allocator.allocate()
        if added in genome.modules:
            raise ValueError("Module id reuse")
        genome.modules[added] = Module.create(rng)
        source = rng.choice(ids + list(INPUTS))
        genome.ff[added] = {source: rng.uniform(-1.2, 1.2)}
        if rng.random() < 0.5:
            genome.rec[added] = {added: rng.uniform(0.4, 1.0)}
        out = rng.choice(OUTPUTS)
        genome.ff.setdefault(out, {})[added] = rng.uniform(-1.2, 1.2)
    elif kind == "remove_module" and len(ids) > 4:
        removed = rng.choice(ids)
        del genome.modules[removed]
        for mapping in (genome.ff, genome.rec):
            mapping.pop(removed, None)
            for edges in mapping.values():
                edges.pop(removed, None)
        for out in OUTPUTS:
            if not genome.ff.get(out):
                source = rng.choice(sorted(genome.modules, key=node_key))
                genome.ff[out] = {source: rng.uniform(-1.0, 1.0)}
    elif kind == "add_ff":
        target = rng.choice(ids + list(OUTPUTS))
        source = rng.choice(ids + list(INPUTS))
        if source not in genome.ff.get(target, {}):
            add_ff_if_acyclic(genome, source, target, rng.uniform(-1.2, 1.2))
    elif kind == "add_rec":
        target = rng.choice(ids + list(OUTPUTS))
        source = rng.choice(ids + list(OUTPUTS))
        if source not in genome.rec.get(target, {}):
            genome.rec.setdefault(target, {})[source] = rng.uniform(-1.2, 1.2)
    elif kind == "remove_edge":
        mapping = genome.rec if rng.random() < 0.4 else genome.ff
        targets = sorted((k for k, v in mapping.items() if len(v) > 1), key=node_key)
        if targets:
            target = rng.choice(targets)
            source = rng.choice(sorted(mapping[target], key=node_key))
            del mapping[target][source]
    elif kind not in kinds:
        raise ValueError(kind)
    genome.validate()
    return {"kind": kind, "added_module": added, "before": before, "after": genome.digest}


def crossover(first, second, rng, fitness1, fitness2, slot1=0, slot2=1):
    if (fitness2, -slot2) > (fitness1, -slot1):
        first, second = second, first
    child = first.clone()
    for node in sorted(set(first.modules) & set(second.modules), key=node_key):
        if rng.random() < 0.5:
            child.modules[node] = second.modules[node]
    for target in sorted(second.ff, key=node_key):
        if target not in child.modules and target not in OUTPUTS:
            continue
        for source in sorted(second.ff[target], key=node_key):
            if source not in INPUTS and source not in child.modules:
                continue
            weight = second.ff[target][source]
            if source in child.ff.get(target, {}):
                if rng.random() < 0.5:
                    child.ff[target][source] = weight
            elif rng.random() < 0.3:
                add_ff_if_acyclic(child, source, target, weight)
    for target in sorted(second.rec, key=node_key):
        if target not in child.modules and target not in OUTPUTS:
            continue
        for source in sorted(second.rec[target], key=node_key):
            if source not in child.modules and source not in OUTPUTS:
                continue
            chance = 0.5 if source in child.rec.get(target, {}) else 0.35
            if rng.random() < chance:
                child.rec.setdefault(target, {})[source] = second.rec[target][source]
    child.validate()
    return child


class NetworkPolicy:
    def __init__(self, genome):
        genome.validate()
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
        choices = [i for i, p in enumerate(probs) if math.isclose(p, best, rel_tol=1e-9, abs_tol=1e-9)]
        return choices[0] if len(choices) == 1 else rng.choice(choices)


class BatchNetwork:
    """Independent episodes in NumPy columns; same ordered graph operations."""

    def __init__(self, genome, count):
        import numpy as np
        self.policy = NetworkPolicy(genome)
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
