"""사전 지정한 시드 단위 통계. 후보 선발에는 테스트 수치를 쓰지 않는다."""

import math
import numpy as np
from pathlib import Path


def sign_p(positive, n=10):
    if type(positive) is not int or not 0 <= positive <= n:
        raise ValueError("Invalid positive seed count")
    return sum(math.comb(n, count) for count in range(positive, n + 1)) / 2**n


def holm(p_values, alpha=0.05):
    ordered = sorted(enumerate(p_values), key=lambda item: (item[1], item[0]))
    passed, still_passing = [False] * len(p_values), True
    for rank, (index, value) in enumerate(ordered):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid p value")
        still_passing = still_passing and value <= alpha / (len(p_values) - rank)
        passed[index] = still_passing
    return passed


def paired_summary(differences, draws):
    values = np.asarray(differences, dtype=float)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError("Ten finite paired seed differences required")
    ci = np.percentile(values[draws].mean(axis=1), [2.5, 97.5], method="linear").tolist()
    positive = int((values > 0).sum())
    return {"differences": values.tolist(), "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "ci95": ci, "positive_seeds": positive, "sign_p": sign_p(positive)}


def primary_decision(rates):
    if set(rates) != {"A0", "A1", "A2"}:
        raise ValueError("All three conditions required")
    arrays = {name: np.asarray(values, dtype=float) for name, values in rates.items()}
    if any(a.shape != (10,) or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any() for a in arrays.values()):
        raise ValueError("Each condition requires ten finite success rates")
    draws = np.random.default_rng(72004).integers(0, 10, size=(10000, 10))
    results = {name: paired_summary(arrays[name] - arrays["A0"], draws) for name in ("A1", "A2")}
    corrected = holm([results[name]["sign_p"] for name in ("A1", "A2")])
    for name, passed in zip(("A1", "A2"), corrected):
        row = results[name]
        row["holm_pass"] = passed
        row["progress_gate"] = row["mean"] >= 0.10 and row["positive_seeds"] >= 8 and passed
    selected = next((name for name in ("A1", "A2") if results[name]["progress_gate"]), None)
    return {"primary": results, "confirmation_condition": selected,
            "ci_interpretation": "individual descriptive intervals, not simultaneous intervals",
            "test_interpretation": "probability of positive seed difference, not mean effect"}


def diagnostic_summary(records):
    total = len(records)
    failures = sum(not row["success"] for row in records)
    repeated = sum(not row["success"] and len(set(row["actions"][-6:])) == 1 for row in records)
    change_correct = change_total = mirror_correct = mirror_switch = mirror_total = 0
    for row in records:
        observations = row.get("observations", [])
        for index, step in enumerate(observations):
            if index and row["rewards_cents"][index - 1] == 99:
                change_total += 1
                change_correct += step["action"] == step["correct_action"]
            if step["mirror"] is not None:
                mirror_total += 1
                mirror_correct += step["mirror"]["both_correct"]
                mirror_switch += step["mirror"]["action_switched"]
    ratio = lambda num, den: num / den if den else None
    return {"episodes": total, "failures": failures, "last_six_repeated_failures": repeated,
            "repeated_fraction_all": ratio(repeated, total), "repeated_fraction_failures": ratio(repeated, failures),
            "goal_change": {"correct": change_correct, "total": change_total, "fraction": ratio(change_correct, change_total)},
            "mirror": {"both_correct": mirror_correct, "switched": mirror_switch, "eligible": mirror_total,
                       "both_correct_fraction": ratio(mirror_correct, mirror_total), "switch_fraction": ratio(mirror_switch, mirror_total)},
            "failed_collect_per_episode": ratio(sum(row["failed_collect"] for row in records), total),
            "walls_per_episode": ratio(sum(row["walls"] for row in records), total)}


def summarize_results(root):
    from .protocol import load
    from .search import directory
    from .storage import read_json
    root = Path(root)
    config = load(root)
    if config["phase"] not in ("main", "confirmation") or len(config["seeds"]) != 10:
        raise ValueError("Analysis requires ten completed production seeds")
    baseline = read_json(root / "baseline_results.json")
    rates, probes, rows = {}, {}, []
    for condition in config["conditions"]:
        rates[condition], probes[condition] = {}, []
        for method in config["methods"]:
            values = []
            for seed in config["seeds"]:
                path = directory(root, condition, method, seed)
                tested = read_json(path / "test.json.gz")
                probe = read_json(path / "probe.json.gz")
                metrics = tested["roles"]["best"]["metrics"]
                values.append(metrics["success_rate"])
                rows.append({"condition": condition, "method": method, "seed": seed, "roles": {
                    role: tested["roles"][role]["metrics"] for role in ("best", "initial", "final")},
                    "probe_rate": probe["both_correct_rate"]})
                if method == "evolution":
                    probes[condition].append(probe["both_correct_rate"])
            rates[condition][method] = values
    draws = np.random.default_rng(72004).integers(0, 10, size=(10000, 10))
    if config["phase"] == "main":
        decision = primary_decision({c: rates[c]["evolution"] for c in config["conditions"]})
    else:
        selected = next(c for c in config["conditions"] if c != "A0")
        row = paired_summary(np.asarray(rates[selected]["evolution"]) - rates["A0"]["evolution"], draws)
        row["progress_gate"] = row["mean"] >= 0.10 and row["positive_seeds"] >= 8 and row["sign_p"] <= 0.05
        decision = {"condition": selected, "confirmation": row}
    secondary = {}
    for condition in config["conditions"]:
        evo = np.asarray(rates[condition]["evolution"])
        capability = paired_summary(evo - baseline["blind"]["success_rate"], draws)
        advantage = paired_summary(evo - rates[condition]["random"], draws)
        for item in (capability, advantage):
            item["exploratory_gate"] = item["mean"] >= 0.10 and item["ci95"][0] > 0 and item["positive_seeds"] >= 8
        passing = sum(value >= 0.8 for value in probes[condition])
        capability.update(probe_passing_seeds=passing, capability_gate=capability["exploratory_gate"] and passing >= 8)
        secondary[condition] = {"capability": capability, "evolution_advantage": advantage}
    return {"protocol_digest": config["protocol_digest"], "phase": config["phase"], "rows": rows,
            "rates": rates, "decision": decision, "secondary_exploratory_unadjusted": secondary,
            "baselines": baseline, "benchmark_previously_observed": True}
