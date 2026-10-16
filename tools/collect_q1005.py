"""완료된 Q1-005 기록만 집계한다. 탐색·평가·재선발은 수행하지 않는다."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "studies/interaction"))
from evosim_q1005_v01.protocol import load, RESULTS
from evosim_q1005_v01.search import directory, attempt_accounting
from evosim_interaction_v02.storage import read_json, file_digest


def distribution(values):
    observed = [v for v in values if v is not None]
    return {"mean": statistics.mean(observed) if observed else None,
            "median": statistics.median(observed) if observed else None,
            "sd": statistics.stdev(observed) if len(observed) > 1 else None,
            "minimum": min(observed) if observed else None, "maximum": max(observed) if observed else None,
            "observed_seeds": len(observed), "missing_seeds": len(values) - len(observed), "values": values}


def collect(phase):
    root = RESULTS / phase
    config, summary = load(root), read_json(root / "summary.json")
    audits = {name: read_json(root / (name + ".json")) for name in ("search_audit", "test_audit")}
    for audit in audits.values():
        if not audit["passed"] or len(audit["jobs"]) != len(config["jobs"]) or not all(r["passed"] for r in audit["jobs"]):
            raise ValueError("All registered jobs must pass audits")
    if (root / "failures.json").exists():
        raise ValueError("Recorded failure needs separate review")
    counts, groups, rankings, times, starts, histories = Counter(), {}, {}, [], [], {}
    for job in config["jobs"]:
        path = directory(root, **job)
        result, checkpoint, tested = (read_json(path / name) for name in ("result.json", "checkpoint.json", "test.json.gz"))
        if checkpoint["next_round"] != config["rounds"] or not checkpoint["result_digest"]:
            raise ValueError("Incomplete checkpoint")
        for key in ("training_slots", "validation_slots", "logical_episode_slots"):
            counts["search_" + key] += result[key]
        for scope, target in (("search", path), ("test", path / "test_work"), ("probe", path / "probe_work")):
            for key, value in attempt_accounting(target).items():
                counts[scope + "_" + key] += value
        counts["test_logical_episode_slots"] += tested["logical_episode_slots"]
        counts["test_unique_role_episode_calls"] += tested["unique_role_episode_calls"]
        counts["probe_pairs"] += tested["probe_pairs"]
        counts["probe_single_step_actions"] += tested["probe_single_step_actions"]
        counts["cached_role_slots"] += sum(row["cache_reused"] for row in tested["roles"].values())
        group = groups.setdefault(job["condition"] + "/" + job["method"], [])
        group.append(next(row for row in summary["rows"] if all(row[k] == v for k, v in job.items())))
        rank = rankings.setdefault(job["condition"] + "/" + job["method"], Counter())
        histories.setdefault(job["condition"] + "/" + job["method"], []).append(result["history"])
        for row in result["history"]:
            rank.update({key: int(value) for key, value in row.get("ranking", {}).items()})
        times.append(checkpoint["elapsed_seconds"])
        starts.extend(read_json(p)["started_utc"] for p in (path / "attempts").glob("*.json"))
    for scope in ("blind", "baseline"):
        for key, value in attempt_accounting(root / (scope + "_work")).items():
            counts[scope + "_" + key] += value
    counts["search_training_episode_slots"] = counts["search_training_slots"] * config["training_batch"]
    counts["search_validation_episode_slots"] = counts["search_logical_episode_slots"] - counts["search_training_episode_slots"]
    counts["search_genomes_audited"] = sum(r["genomes_checked"] for r in audits["search_audit"]["jobs"])
    counts["search_episode_records_replayed"] = sum(r["episode_records_replayed"] for r in audits["search_audit"]["jobs"])
    counts["test_episode_records_replayed"] = sum(r["episode_records_replayed"] for r in audits["test_audit"]["jobs"])
    counts["best_diagnostic_episode_records_replayed"] = len(config["jobs"]) * 1512
    descriptions = {}
    for group, rows in groups.items():
        descriptions[group] = {role: {metric: distribution([r["roles"][role][metric] for r in rows])
            for metric in rows[0]["roles"][role]} for role in ("best", "initial", "final")}
        descriptions[group]["probe_rate"] = distribution([r["probe_rate"] for r in rows])
    last = datetime.fromtimestamp((root / "summary.json").stat().st_mtime, timezone.utc)
    curves = {}
    for group, runs in histories.items():
        curves[group] = []
        for generation in range(config["rounds"]):
            rows = [run[generation] for run in runs]
            curve = {"round": generation, "validation_performed": rows[0]["validation_performed"]}
            for key in ("training_mean_reward", "training_mean_score", "training_mean_success", "mean_modules"):
                curve[key] = distribution([row[key] for row in rows])
            curve["best_validation_success"] = distribution([row["best_validation"]["success_rate"] for row in rows])
            curves[group].append(curve)
    return {"phase": phase, "source_digest": config["source_digest"], "protocol_digest": config["protocol_digest"],
            "jobs": len(config["jobs"]), "counts": dict(counts), "groups": descriptions, "learning_curves": curves,
            "ranking_history_totals": {k: dict(v) for k, v in rankings.items()},
            "decision": summary["decision"], "first_search_attempt_utc": min(starts),
            "summary_written_utc": last.isoformat(), "first_attempt_to_summary_seconds": (last - datetime.fromisoformat(min(starts))).total_seconds(),
            "search_worker_seconds_sum": sum(times),
            "artifact_sha256": {name: file_digest(root / (name + ".json")) for name in
                ("protocol", "datasets", "candidate_lock", "summary", "search_audit", "test_audit")},
            "scope": "completed records only; action replay is separate from policy episode calls; pooled visited states are descriptive"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("main", "confirmation"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Completed aggregation is immutable")
    result = collect(args.phase)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"phase": args.phase, "jobs": result["jobs"], "counts": result["counts"]}, ensure_ascii=False))
