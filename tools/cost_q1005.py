"""본 결과를 열기 전 훈련 목록만으로 batch 크기별 계산·저장 비용을 측정한다."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "studies/interaction"))
from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.network import Genome, ModuleIds, mutate
from evosim_interaction_v02.storage import read_json, write_once, file_digest
from evosim_q1005_v01.protocol import load, RESULTS
from evosim_q1005_v01.model import Treatment, NETWORK, CONDITIONS
from evosim_q1005_v01.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        raise ValueError("Use a new in-workspace cost report path")
    config = load(RESULTS / "pilot")
    if not read_json(RESULTS / "pilot/cost_report.json")["all_jobs_complete"]:
        raise ValueError("Training-only pilot not complete")
    data = read_json(RESULTS / "pilot/R0S0/evolution_911/result.json")
    original = Genome.from_dict(data["initial_population"][0], NETWORK)
    larger = original.clone()
    rng, allocator = random.Random(27005), ModuleIds()
    for _ in range(5):
        mutate(larger, rng, allocator, NETWORK, forced_kind="add_module")
    if len(larger.modules) != 9:
        raise ValueError("Synthetic cost template must have nine modules")
    rows = []
    for name, genome in (("initial4", original), ("synthetic9", larger)):
        for condition in CONDITIONS:
            for kind, episodes, repeats, training in (("training", make_split().train[:24], 1, True),
                                                       ("validation_shape", make_split().train, 1, False),
                                                       ("test_shape", make_split().train, 2, False)):
                purpose = "train:cost24" if training else "cost:" + kind
                started = time.perf_counter()
                metrics, records = evaluate(genome, episodes, 27005, purpose, Treatment(condition), repeats=repeats, training=training)
                kernel_seconds = time.perf_counter() - started
                path = ROOT / "dist/q1005_cost_probe" / f"{name}_{condition}_{kind}.json.gz"
                started = time.perf_counter()
                write_once(path, {"training_data_only": True, "metrics": metrics, "records": records})
                storage_seconds = time.perf_counter() - started
                rows.append({"template": name, "modules": len(genome.modules), "condition": condition,
                             "shape": kind, "episode_calls": len(records), "kernel_seconds": kernel_seconds,
                             "storage_seconds": storage_seconds, "gzip_bytes": path.stat().st_size,
                             "artifact": path.relative_to(ROOT).as_posix(), "sha256": file_digest(path)})
    projections = {}
    for name in ("initial4", "synthetic9"):
        average = lambda kind, field: sum(r[field] for r in rows if r["template"] == name and r["shape"] == kind) / 4
        search = (200000 * (average("training", "kernel_seconds") + average("training", "storage_seconds"))
                  + 22000 * (average("validation_shape", "kernel_seconds") + average("validation_shape", "storage_seconds"))) / 2
        test = 300 * (average("test_shape", "kernel_seconds") + average("test_shape", "storage_seconds"))
        projections[name] = {"two_worker_search_seconds": search, "serial_uncached_three_role_test_seconds": test,
                             "partial_total_minutes": (search + test) / 60,
                             "search_gzip_bytes_sample_model": 200000 * average("training", "gzip_bytes") + 22000 * average("validation_shape", "gzip_bytes")}
    minimum = min(p["partial_total_minutes"] for p in projections.values())
    maximum = max(p["partial_total_minutes"] for p in projections.values())
    record = {"source_digest": config["source_digest"], "plan_digest": config["plan_digest"],
              "scope": "training-list cost samples only; no validation/test episode IDs and no treatment selection",
              "additional_training_list_episode_calls": sum(r["episode_calls"] for r in rows),
              "cost_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "samples": rows, "projections": projections,
              "rough_main_planning_minutes": [minimum * 1.3, maximum * 3],
              "limits": "not an ETA or confidence interval; excludes audit/checkpoint overhead; synthetic templates and machine load may differ"}
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: record[k] for k in ("additional_training_list_episode_calls", "projections", "rough_main_planning_minutes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
