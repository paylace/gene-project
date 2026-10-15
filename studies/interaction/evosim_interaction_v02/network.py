"""조건을 운반하는 유전체와 유전 연산. v0.1 완료 소스를 수정하지 않는다."""

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

from .settings import OPS, Settings

INPUTS = tuple(f"in_{i}" for i in range(4))
OUTPUTS = tuple(f"out_{i}" for i in range(3))
KINDS = ("weight", "add_module", "remove_module", "add_ff", "add_rec", "remove_edge")


def node_key(node):
    if node in INPUTS:
        return 0, int(node[3:])
    if isinstance(node, str) and re.fullmatch(r"m(0|[1-9][0-9]*)", node):
        return 1, int(node[1:])
    if node in OUTPUTS:
        return 2, int(node[4:])
    raise ValueError(f"Invalid node: {node}")


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class Module:
    op: str
    bias: float
    memory_decay: float

    @classmethod
    def create(cls, rng, settings):
        p = settings.parameters
        return cls(rng.choice(OPS), rng.uniform(*p["module_bias_initial"]), rng.uniform(*p["module_decay_initial"]))

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

    def mutate(self, rng, settings, rejected):
        p = settings.parameters
        op, bias, decay = self.op, self.bias, self.memory_decay
        if rng.random() < p["module_op_change_probability"]:
            proposal = rng.choice(OPS)
            if proposal in settings.allowed_ops:
                op = proposal
            else:
                rejected.append({"reason": "condition_forbidden", "proposal": "module_op", "op": proposal})
        if rng.random() < p["module_bias_change_probability"]:
            bias = max(-2.0, min(2.0, bias + rng.gauss(0, p["module_bias_change_sigma"])))
        if rng.random() < p["module_decay_change_probability"]:
            decay = max(0.0, min(0.95, decay + rng.gauss(0, p["module_decay_change_sigma"])))
        return Module(op, bias, decay)


@dataclass
class ModuleIds:
    next_id: int = 4

    def validate(self, genomes=()):
        if type(self.next_id) is not int or self.next_id < 4:
            raise ValueError("Invalid next module id")
        maximum = max((node_key(node)[1] for genome in genomes for node in genome.modules), default=3)
        if self.next_id <= maximum:
            raise ValueError("Module id allocator is behind the population")

    def allocate(self):
        self.validate()
        result = f"m{self.next_id}"
        self.next_id += 1
        return result


@dataclass
class Genome:
    modules: dict[str, Module]
    ff: dict[str, dict[str, float]]
    rec: dict[str, dict[str, float]]
    settings: Settings

    def clone(self):
        return copy.deepcopy(self)

    def to_dict(self):
        return {"modules": {key: asdict(value) for key, value in self.modules.items()},
                "ff": copy.deepcopy(self.ff), "rec": copy.deepcopy(self.rec)}

    @classmethod
    def from_dict(cls, data, settings):
        if not isinstance(data, dict) or set(data) != {"modules", "ff", "rec"}:
            raise ValueError("Invalid genome schema")
        try:
            result = cls({key: Module(**value) for key, value in data["modules"].items()},
                         copy.deepcopy(data["ff"]), copy.deepcopy(data["rec"]), settings)
            result.validate()
        except (TypeError, AttributeError) as error:
            raise ValueError("Invalid genome data") from error
        return result

    @property
    def digest(self):
        data = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(data.encode()).hexdigest()

    def export(self):
        self.validate()
        return {"schema": 1, "settings": self.settings.to_dict(), "settings_digest": self.settings.digest,
                "genome": self.to_dict(), "genome_digest": self.digest}

    @classmethod
    def restore(cls, snapshot, settings):
        if not isinstance(snapshot, dict) or set(snapshot) != {
            "schema", "settings", "settings_digest", "genome", "genome_digest"
        } or type(snapshot["schema"]) is not int or snapshot["schema"] != 1:
            raise ValueError("Invalid snapshot schema")
        if Settings.from_dict(snapshot["settings"]) != settings or snapshot["settings_digest"] != settings.digest:
            raise ValueError("Snapshot condition/settings mismatch")
        result = cls.from_dict(snapshot["genome"], settings)
        if result.digest != snapshot["genome_digest"]:
            raise ValueError("Snapshot genome digest mismatch")
        return result

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

    def validate(self, settings=None):
        if not isinstance(self.settings, Settings) or (settings is not None and self.settings != settings):
            raise ValueError("Genome condition/settings mismatch")
        if not isinstance(self.modules, dict) or not 1 <= len(self.modules) <= self.settings.parameters["max_modules"]:
            raise ValueError("Invalid module count")
        for node, module in self.modules.items():
            if node_key(node)[0] != 1 or not isinstance(module, Module):
                raise ValueError("Invalid module id/type")
            if module.op not in self.settings.allowed_ops or not all(
                finite_number(value) for value in (module.bias, module.memory_decay)
            ) or not 0 <= module.memory_decay <= 0.95:
                raise ValueError("Invalid/forbidden module")
        targets = set(self.modules) | set(OUTPUTS)
        for mapping, sources in ((self.ff, set(self.modules) | set(INPUTS)),
                                 (self.rec, set(self.modules) | set(OUTPUTS))):
            if not isinstance(mapping, dict):
                raise ValueError("Invalid edge mapping")
            for target, edges in mapping.items():
                if target not in targets or not isinstance(edges, dict) or not set(edges) <= sources:
                    raise ValueError("Dangling or invalid edge")
                if not all(finite_number(weight) for weight in edges.values()):
                    raise ValueError("Nonfinite/invalid weight")
        if self.settings.condition == "A2" and self.rec != {}:
            raise ValueError("A2 requires rec == {}, including no empty dictionaries")
        if any(not self.settings.allows_rec(source, target) for target, edges in self.rec.items() for source in edges):
            raise ValueError("Forbidden recurrent edge")
        self.topological_order()


def initial_genome(rng, settings):
    """모든 원형 추첨을 마친 뒤에만 조건 변환. 변환은 초기 생성에만 사용한다."""
    p = settings.parameters
    modules = {f"m{i}": Module.create(rng, settings) for i in range(p["initial_modules"])}
    ids = list(modules)
    ff, rec = {}, {}
    for i, node in enumerate(ids):
        ff[node] = {src: rng.uniform(*p["input_weight_initial"]) for src in INPUTS}
        if i:
            ff[node][ids[i - 1]] = rng.uniform(*p["chain_weight_initial"])
        if rng.random() < p["initial_self_rec_probability"]:
            rec[node] = {node: rng.uniform(*p["initial_self_rec_weight"])}
    for i, out in enumerate(OUTPUTS):
        ff[out] = {ids[i]: rng.uniform(*p["output_primary_weight"]),
                   ids[(i + 1) % 4]: rng.uniform(*p["output_aux_weight"])}
        rec[out] = {ids[i]: rng.uniform(*p["output_module_rec_weight"]), out: rng.uniform(*p["output_self_rec_weight"])}
    if settings.condition == "A1":
        for out in OUTPUTS:
            rec[out].pop(out)
    elif settings.condition == "A2":
        rec = {}
        modules = {node: module if module.op in settings.allowed_ops else Module("tanh", module.bias, module.memory_decay)
                   for node, module in modules.items()}
    result = Genome(modules, ff, rec, settings)
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


def mutate(genome, rng, allocator, settings, forced_kind=None):
    genome.validate(settings)
    allocator.validate((genome,))
    p = settings.parameters
    kind = forced_kind or rng.choices(KINDS, p["mutation_weights"], k=1)[0]
    before = genome.digest
    added, rejected = None, []
    ids = sorted(genome.modules, key=node_key)
    if kind == "weight":
        for node in rng.sample(ids, min(2, len(ids))):
            genome.modules[node] = genome.modules[node].mutate(rng, settings, rejected)
        for mapping in (genome.ff, genome.rec):
            for target in sorted(mapping, key=node_key):
                for src in sorted(mapping[target], key=node_key):
                    if rng.random() < p["weight_change_probability"]:
                        mapping[target][src] = max(-3.0, min(3.0, mapping[target][src] + rng.gauss(0, p["weight_change_sigma"])))
    elif kind == "add_module" and len(ids) < p["max_modules"]:
        added = allocator.allocate()
        if added in genome.modules:
            raise ValueError("Module id reuse")
        module = Module.create(rng, settings)
        if module.op not in settings.allowed_ops:
            rejected.append({"reason": "condition_forbidden", "proposal": "new_module_op", "op": module.op})
            module = Module("tanh", module.bias, module.memory_decay)
        genome.modules[added] = module
        source = rng.choice(ids + list(INPUTS))
        genome.ff[added] = {source: rng.uniform(*p["new_edge_weight"])}
        if rng.random() < p["new_module_self_rec_probability"]:
            weight = rng.uniform(*p["new_module_self_rec_weight"])
            if settings.allows_rec(added, added):
                genome.rec[added] = {added: weight}
            else:
                rejected.append({"reason": "condition_forbidden", "proposal": "new_module_rec"})
        out = rng.choice(OUTPUTS)
        genome.ff.setdefault(out, {})[added] = rng.uniform(*p["new_edge_weight"])
    elif kind == "remove_module" and len(ids) > p["initial_modules"]:
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
        if source in genome.ff.get(target, {}):
            rejected.append({"reason": "duplicate_edge", "proposal": "add_ff"})
        elif not add_ff_if_acyclic(genome, source, target, rng.uniform(*p["new_edge_weight"])):
            rejected.append({"reason": "ff_cycle", "proposal": "add_ff"})
    elif kind == "add_rec":
        target = rng.choice(ids + list(OUTPUTS))
        source = rng.choice(ids + list(OUTPUTS))
        if source in genome.rec.get(target, {}):
            rejected.append({"reason": "duplicate_edge", "proposal": "add_rec"})
        else:
            weight = rng.uniform(*p["new_edge_weight"])
            if settings.allows_rec(source, target):
                genome.rec.setdefault(target, {})[source] = weight
            else:
                rejected.append({"reason": "condition_forbidden", "proposal": "add_rec", "source": source, "target": target})
    elif kind == "remove_edge":
        mapping = genome.rec if rng.random() < p["remove_edge_rec_probability"] else genome.ff
        targets = sorted((key for key, value in mapping.items() if len(value) > 1), key=node_key)
        if targets:
            target = rng.choice(targets)
            source = rng.choice(sorted(mapping[target], key=node_key))
            del mapping[target][source]
        else:
            rejected.append({"reason": "no_removable_target", "proposal": "remove_edge"})
    elif kind not in KINDS:
        raise ValueError(kind)
    else:
        rejected.append({"reason": "module_limit", "proposal": kind})
    genome.validate(settings)
    after = genome.digest
    if before == after and not rejected:
        rejected.append({"reason": "no_parameter_change", "proposal": kind})
    return {"kind": kind, "added_module": added, "before": before, "after": after,
            "changed": before != after, "proposal_rejected": any(r["reason"] == "condition_forbidden" for r in rejected),
            "rejections": rejected}


def crossover(first, second, rng, fitness1, fitness2, settings, slot1=0, slot2=1):
    first.validate(settings)
    second.validate(settings)
    if not all(finite_number(value) for value in (fitness1, fitness2)):
        raise ValueError("Nonfinite fitness")
    p = settings.parameters
    if (fitness2, -slot2) > (fitness1, -slot1):
        first, second = second, first
    child = first.clone()
    for node in sorted(set(first.modules) & set(second.modules), key=node_key):
        if rng.random() < p["matching_gene_second_probability"]:
            child.modules[node] = second.modules[node]
    for target in sorted(second.ff, key=node_key):
        if target not in child.modules and target not in OUTPUTS:
            continue
        for source in sorted(second.ff[target], key=node_key):
            if source not in INPUTS and source not in child.modules:
                continue
            weight = second.ff[target][source]
            if source in child.ff.get(target, {}):
                if rng.random() < p["matching_gene_second_probability"]:
                    child.ff[target][source] = weight
            elif rng.random() < p["second_parent_ff_probability"]:
                add_ff_if_acyclic(child, source, target, weight)
    for target in sorted(second.rec, key=node_key):
        if target not in child.modules and target not in OUTPUTS:
            continue
        for source in sorted(second.rec[target], key=node_key):
            if source not in child.modules and source not in OUTPUTS:
                continue
            chance = p["matching_gene_second_probability"] if source in child.rec.get(target, {}) else p["second_parent_rec_probability"]
            if rng.random() < chance:
                child.rec.setdefault(target, {})[source] = second.rec[target][source]
    child.validate(settings)
    return child
