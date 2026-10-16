"""완료한 Q1-004 기록을 보고용으로 집계한다. 탐색·평가는 실행하지 않는다."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "studies/interaction"))

from evosim_interaction_v02.protocol import load
from evosim_interaction_v02.search import attempt_accounting
from evosim_interaction_v02.storage import file_digest, read_json


def distribution(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values),
            "min": min(values), "max": max(values), "values": values}


def collect(phase):
    root = ROOT / "studies/interaction/results_v02" / phase
    config = load(root)
    summary = read_json(root / "summary.json")
    search_audit = read_json(root / "search_audit.json")
    test_audit = read_json(root / "test_audit.json")
    expected = len(config["conditions"]) * len(config["methods"]) * len(config["seeds"])
    if (len(search_audit["jobs"]) != expected or len(test_audit["jobs"]) != expected
            or not all(j["passed"] for j in search_audit["jobs"] + test_audit["jobs"])):
        raise ValueError("All production audits must pass before report aggregation")
    if (root / "failures.json").exists():
        raise ValueError("Search failure artifact requires explicit review")
    counts = {k: 0 for k in ("training_candidate_slots", "validation_candidate_slots",
              "search_logical_episode_slots", "search_known_episode_calls", "unknown_interrupted_attempts",
              "unknown_interrupted_calls_upper_bound", "test_logical_episode_slots", "test_actual_episode_calls",
              "cached_role_slots", "probe_pairs", "probe_single_step_actions")}
    times, first, groups, diagnostics, mutations, initialization = [], [], {}, {}, {}, {}
    for condition in config["conditions"]:
        for method in config["methods"]:
            selected, observed = [], []
            proposal_kinds, rejections, init_ops = Counter(), Counter(), Counter()
            proposal_count = changed = forbidden = 0
            for seed in config["seeds"]:
                path = root / condition / f"{method}_{seed}"
                result = read_json(path / "result.json")
                for genome in result["initial_population"]:
                    init_ops.update(module["op"] for module in genome["modules"].values())
                checkpoint = read_json(path / "checkpoint.json")
                accounting = attempt_accounting(path)
                tested = read_json(path / "test.json.gz")
                probe = read_json(path / "probe.json.gz")
                if checkpoint["next_round"] != config["rounds"] or not checkpoint["result_digest"]:
                    raise ValueError("Incomplete checkpoint")
                counts["training_candidate_slots"] += result["training_slots"]
                counts["validation_candidate_slots"] += result["validation_slots"]
                counts["search_logical_episode_slots"] += result["logical_episode_slots"]
                counts["search_known_episode_calls"] += accounting["known_completed_episode_calls"]
                counts["unknown_interrupted_attempts"] += accounting["interrupted_attempts"]
                counts["unknown_interrupted_calls_upper_bound"] += accounting["unknown_interrupted_calls_upper_bound"]
                counts["test_logical_episode_slots"] += tested["logical_episode_slots"]
                counts["test_actual_episode_calls"] += tested["actual_episode_calls"]
                counts["cached_role_slots"] += sum(r["cache_reused"] for r in tested["roles"].values())
                counts["probe_pairs"] += len(probe["records"])
                counts["probe_single_step_actions"] += 2 * len(probe["records"])
                times.append(checkpoint["elapsed_seconds"])
                first.extend(read_json(p)["started_utc"] for p in (path / "attempts").glob("*.json"))
                selected.append({"roles": {r: tested["roles"][r]["metrics"] for r in ("best", "initial", "final")},
                                 "probe_rate": probe["both_correct_rate"]})
                if phase == "main":
                    observed.append(read_json(path / "diagnostics.json.gz")["summary"])
                if method == "evolution":
                    for round_path in sorted(path.glob("round_*.json.gz")):
                        for row in read_json(round_path)["candidates"]:
                            mutation = row["ancestry"].get("mutation")
                            if mutation is None:
                                continue
                            proposal_count += 1
                            proposal_kinds[mutation["kind"]] += 1
                            changed += mutation["changed"]
                            forbidden += mutation["proposal_rejected"]
                            rejections.update(item["reason"] for item in mutation["rejections"])
            group = {}
            for role in ("best", "initial", "final"):
                group[role] = {key: distribution([s["roles"][role][key] for s in selected])
                               for key in ("success_rate", "mean_reward", "mean_collected", "mean_failed_collect", "mean_walls")}
            group["probe_rate"] = distribution([s["probe_rate"] for s in selected])
            groups[f"{condition}/{method}"] = group
            initialization[f"{condition}/{method}"] = dict(init_ops)
            if method == "evolution":
                mutations[condition] = {"proposals": proposal_count, "changed": changed,
                    "unchanged": proposal_count - changed, "forbidden_proposal_attempts": forbidden,
                    "kinds": dict(proposal_kinds), "rejection_reasons": dict(rejections),
                    "changed_fraction": changed / proposal_count if proposal_count else None,
                    "forbidden_fraction": forbidden / proposal_count if proposal_count else None}
            if observed:
                totals = {key: sum(s[key] for s in observed) for key in ("episodes", "failures", "last_six_repeated_failures")}
                for kind, num_keys, denominator in (("goal_change", ("correct",), "total"),
                                                   ("mirror", ("both_correct", "switched"), "eligible")):
                    row = {k: sum(s[kind][k] for s in observed) for k in (*num_keys, denominator)}
                    for k in num_keys:
                        row[k + "_fraction"] = row[k] / row[denominator] if row[denominator] else None
                    totals[kind] = row
                totals["repeated_fraction_failures"] = totals["last_six_repeated_failures"] / totals["failures"] if totals["failures"] else None
                diagnostics[f"{condition}/{method}"] = totals
    counts["search_training_episode_slots"] = counts["training_candidate_slots"] * config["training_batch"]
    counts["search_validation_episode_slots"] = counts["search_logical_episode_slots"] - counts["search_training_episode_slots"]
    counts["test_episode_records_replayed"] = sum(j["episode_records_replayed"] for j in test_audit["jobs"])
    counts["search_genomes_audited"] = sum(j["genomes_checked"] for j in search_audit["jobs"])
    counts["observer_mirror_pairs"] = sum(row["mirror"]["eligible"] for row in diagnostics.values())
    counts["observer_additional_single_step_forwards"] = 2 * counts["observer_mirror_pairs"]
    if phase == "main":
        baseline = read_json(root / "baseline_records.json.gz")
        counts["baseline_test_episode_calls"] = sum(len(v) for v in baseline.values())
        counts["blind_validation_episode_calls"] = sum(len(v["records"]) for v in read_json(root / "blind_validation.json.gz")["candidates"])
    else:
        counts["baseline_test_episode_calls"] = 0
        counts["blind_validation_episode_calls"] = 0
    artifacts = {p.name: file_digest(p) for p in (root / name for name in
                 ("protocol.json", "datasets.json", "candidate_lock.json", "search_audit.json", "test_audit.json", "summary.json"))}
    last = datetime.fromtimestamp((root / "summary.json").stat().st_mtime, timezone.utc)
    output = {"phase": phase, "source_digest": config["source_digest"], "protocol_digest": config["protocol_digest"],
              "jobs": expected, "counts": counts, "first_search_attempt_utc": min(first),
              "summary_written_utc": last.isoformat(),
              "first_attempt_to_summary_seconds": (last - datetime.fromisoformat(min(first))).total_seconds(),
              "search_worker_seconds_sum": sum(times), "groups": groups, "diagnostics": diagnostics,
              "mutation_records": mutations, "initial_module_operation_counts": initialization,
              "decision": summary["decision"], "artifact_sha256": artifacts,
              "aggregation_scope": "completed records only; no policy evaluation or reselection",
              "diagnostic_scope": "pooled visited states, not independent statistical samples"}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("main", "confirmation"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.phase)
    if args.output.exists():
        raise FileExistsError("Refuse to overwrite a completed aggregation")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"phase": args.phase, "jobs": result["jobs"], "counts": result["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
