"""20개 짝 시드의 사전 등록 대비와 기술 통계."""

import math
from pathlib import Path
import numpy as np

from evosim_interaction_v02.analysis import holm
from evosim_interaction_v02.storage import read_json
from .protocol import load
from .source import plan
from .search import directory


def sign_p(positive, n=20):
    if type(positive) is not int or not 0 <= positive <= n:
        raise ValueError("Invalid positive count")
    return sum(math.comb(n, count) for count in range(positive, n + 1)) / 2 ** n


def paired_summary(values, draws):
    values = np.asarray(values, dtype=float)
    if values.shape != (20,) or not np.isfinite(values).all():
        raise ValueError("Twenty finite paired differences required")
    ci = np.percentile(values[draws].mean(axis=1), [2.5, 97.5], method="linear").tolist()
    positive = int((values > 0).sum())
    return {"differences": values.tolist(), "mean": float(values.mean()), "median": float(np.median(values)),
            "sd": float(values.std(ddof=1)), "ci95": ci, "positive_seeds": positive,
            "negative_seeds": int((values < 0).sum()), "zero_seeds": int((values == 0).sum()),
            "sign_p": sign_p(positive)}


def primary_decision(rates):
    spec = plan()
    if set(rates) != set(spec["conditions"]):
        raise ValueError("All four treatment results required")
    arrays = {name: np.asarray(values, dtype=float) for name, values in rates.items()}
    if any(x.shape != (20,) or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any() for x in arrays.values()):
        raise ValueError("Each condition requires twenty finite rates in [0,1]")
    draws = np.random.default_rng(72005).integers(0, 20, size=(10000, 20))
    contrasts = spec["primary_contrasts"]
    primary = {name: paired_summary(arrays[name] - arrays["R0S0"], draws) for name in contrasts}
    corrected = holm([primary[name]["sign_p"] for name in contrasts])
    for name, passed in zip(contrasts, corrected):
        row = primary[name]
        row["holm_pass"] = passed
        row["progress_gate"] = row["mean"] >= 0.1 - 1e-12 and row["positive_seeds"] >= 15 and passed
    selected = next((name for name in spec["confirmation_priority"] if primary[name]["progress_gate"]), None)
    interaction = paired_summary(arrays["R1S1"] - arrays["R1S0"] - arrays["R0S1"] + arrays["R0S0"], draws)
    return {"primary": primary, "confirmation_condition": selected, "interaction_exploratory": interaction,
            "ci_scope": "individual descriptive percentile intervals, not simultaneous",
            "test_scope": "positive seed difference probability, not mean difference"}


def summarize(root):
    root = Path(root)
    config = load(root)
    if config["phase"] not in ("main", "confirmation") or len(config["seeds"]) != 20:
        raise ValueError("Twenty-seed completed production cohort required")
    rows, rates, descriptions = [], {}, {}
    for job in config["jobs"]:
        path = directory(root, **job)
        tested = read_json(path / "test.json.gz")
        probe = read_json(path / "probe.json.gz")
        rate = tested["roles"]["best"]["metrics"]["success_rate"]
        rates.setdefault(job["condition"], {}).setdefault(job["method"], []).append(rate)
        rows.append({**job, "roles": {r: tested["roles"][r]["metrics"] for r in ("best", "initial", "final")},
                     "probe_rate": probe["both_correct_rate"], "trajectory": read_json(path / "diagnostics.json")})
    for condition, methods in rates.items():
        for method, values in methods.items():
            descriptions[f"{condition}/{method}"] = {"mean": float(np.mean(values)), "median": float(np.median(values)),
                "sd": float(np.std(values, ddof=1)), "minimum": min(values), "maximum": max(values),
                "at_least_80_percent_seeds": sum(v >= 0.8 for v in values), "rates": values}
    draws = np.random.default_rng(72005).integers(0, 20, size=(10000, 20))
    if config["phase"] == "main":
        decision = primary_decision({c: rates[c]["evolution"] for c in config["conditions"]})
    else:
        selected = next(c for c in config["conditions"] if c != "R0S0")
        row = paired_summary(np.asarray(rates[selected]["evolution"]) - rates["R0S0"]["evolution"], draws)
        row["progress_gate"] = row["mean"] >= 0.1 - 1e-12 and row["positive_seeds"] >= 15 and row["sign_p"] <= 0.05
        decision = {"condition": selected, "confirmation": row}
    baselines = read_json(root / "baseline_results.json")
    secondary = {}
    for condition in config["conditions"]:
        values = np.asarray(rates[condition]["evolution"])
        secondary[condition] = {"evolution_minus_shared_random": paired_summary(values - rates["R0S0"]["random"], draws),
                                "evolution_minus_blind": paired_summary(values - baselines["blind"]["success_rate"], draws)}
    return {"phase": config["phase"], "protocol_digest": config["protocol_digest"], "rows": rows, "rates": rates,
            "descriptions": descriptions, "decision": decision, "secondary_exploratory_unadjusted": secondary,
            "baselines": baselines, "random_control_shared": True, "benchmark_previously_observed": True,
            "immediate_probe_is_distinct_from_rollout_recovery": True}
