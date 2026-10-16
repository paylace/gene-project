"""확인 결과 반전 후 동결 후보·기록을 같은 커널로 재계산하는 기술 점검."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "studies/interaction"))
from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.network import Genome
from evosim_interaction_v02.protocol import load, evaluate_split
from evosim_interaction_v02.settings import Settings
from evosim_interaction_v02.storage import read_json


def main():
    plan = read_json(ROOT / "docs/management/q1_004_replication_check_plan.json")
    output_path = ROOT / "docs/management/q1_004_replication_check.json"
    if output_path.exists():
        raise FileExistsError("Technical replication output already exists")
    rows, calls = [], 0
    for phase, conditions in plan["phases"].items():
        root = ROOT / "studies/interaction/results_v02" / phase
        config = load(root)
        if not (root / "summary.json").exists():
            raise ValueError("Only completed production results can be rechecked")
        if config["conditions"] != conditions:
            raise ValueError("Conditions differ from recorded recheck plan")
        for condition in conditions:
            settings = Settings(condition)
            for seed in config["seeds"]:
                path = root / condition / f"evolution_{seed}"
                selected = read_json(path / "result.json")["best"]
                stored = read_json(path / "test.json.gz")["roles"]["best"]
                genome = Genome.from_dict(selected["genome"], settings)
                metrics, records = evaluate_split(root, genome, make_split().test, seed, "test", repeats=3, observe=False)
                calls += len(records)
                matches = metrics == stored["metrics"] and records == stored["records"]
                rows.append({"phase": phase, "condition": condition, "seed": seed, "digest": genome.digest,
                             "records_match": matches, "episode_calls": len(records),
                             "output_self_weights": {node: genome.rec.get(node, {}).get(node) for node in ("out_0", "out_1", "out_2")},
                             "modules": len(genome.modules)})
                print(f"RECHECK {phase}/{condition}/{seed} match={matches}", flush=True)
    output = {"scope": plan["scope"], "planned_episode_calls": plan["planned_episode_calls"],
              "actual_episode_calls_separate": calls, "jobs": len(rows),
              "all_match": all(row["records_match"] for row in rows), "rows": rows,
              "no_search_or_reselection": True, "no_changes_to_decision_gates": True}
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not output["all_match"] or calls != plan["planned_episode_calls"]:
        raise SystemExit("Technical recheck failed; retain artifacts for review")


if __name__ == "__main__":
    main()
