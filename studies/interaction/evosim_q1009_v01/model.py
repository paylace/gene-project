"""변환·초기화와 수작업 대조. 기존 신경망 계산은 동결 의존성이다."""
import random
import numpy as np
from evosim_interaction_v02.settings import Settings
from evosim_interaction_v02.network import Genome, Module, initial_genome

SETTINGS = Settings("A0")
CONDITIONS = ("C0N0", "C1N0", "C0N1", "C1N1")


def validate_condition(condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")


def observations(raw, condition):
    validate_condition(condition)
    result = np.array(raw, dtype=float, copy=True)
    if condition[1] == "1":
        result[..., :2] = 2 * result[..., :2] - 1
    return result


def transform(genome, condition):
    validate_condition(condition)
    result = genome.clone()
    if condition[3] == "1":
        for output in ("out_0", "out_1", "out_2"):
            result.rec.get(output, {}).pop(output, None)
    result.validate(SETTINGS)
    return result


def initial(seed):
    return initial_genome(random.Random(seed), SETTINGS)


def control(name):
    modules = {f"m{i}": Module("linear", 0.0, 0.5) for i in range(4)}
    if name == "coordinate":
        edges = {"out_0": {"in_0": 2.0, "in_1": -2.0},
                 "out_1": {"in_0": -2.0, "in_1": 2.0}, "out_2": {"m0": 0.5}}
    elif name in ("left", "right", "collect", "zero"):
        action = {"left": 0, "right": 1, "collect": 2, "zero": None}[name]
        edges = {} if action is None else {f"out_{action}": {"m0": 1.0}}
    else:
        raise ValueError("Unknown control")
    result = Genome(modules, edges, {}, SETTINGS)
    result.validate()
    return result
