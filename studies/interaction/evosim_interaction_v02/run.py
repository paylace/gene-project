"""단계별 실행기. 예비는 훈련만 하며, 본 테스트는 전체 후보 잠금 후에 연다."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

from .audit import audit_job
from .protocol import ROOT, prepare
from .search import run_search, directory, attempt_accounting
from .storage import write_json, read_json


def execute(root, config, workers=2, pause_after=None):
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_search, root, c, m, s, pause_after=pause_after): (c, m, s)
                   for c in config["conditions"] for m in config["methods"] for s in config["seeds"]}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
                print(f"DONE {'/'.join(map(str, job))}", flush=True)
            except Exception as error:
                errors.append({"job": job, "error": repr(error)})
                write_json(Path(root) / "failures.json", errors)
    if errors:
        raise RuntimeError("Search failures retained; do not treat them as zero performance")


def pilot_report(root, config, elapsed):
    root = Path(root)
    audits, calls, known = [], 0, 0
    for c in config["conditions"]:
        for m in config["methods"]:
            for s in config["seeds"]:
                audits.append(audit_job(root, c, m, s))
                result = read_json(directory(root, c, m, s) / "result.json")
                calls += result["logical_episode_slots"]
                counts = attempt_accounting(directory(root, c, m, s))
                if counts["interrupted_attempts"]:
                    raise ValueError("Pilot has unknown interrupted work; inspect attempts before cost estimation")
                known += counts["known_completed_episode_calls"]
    # 프로브는 훈련 목록만 사용한다. 실제 예비 탐색 예산과 별도로 계수한다.
    from .datasets import make_split
    from .evaluation import evaluate
    from .network import Genome
    from .settings import Settings
    import gzip
    import json
    costs = []
    for c in config["conditions"]:
        data = read_json(directory(root, c, "evolution", config["seeds"][0]) / "result.json")
        genome = Genome.from_dict(data["best"]["genome"], Settings(c))
        started = time.perf_counter()
        _, records = evaluate(genome, make_split().train[:24], 901, "pilot_observer_cost", genome.settings, observe=True)
        payload = json.dumps(records, separators=(",", ":"), allow_nan=False).encode()
        costs.append({"condition": c, "training_episodes": len(records), "elapsed_seconds": time.perf_counter() - started,
                      "json_bytes": len(payload), "gzip_bytes": len(gzip.compress(payload, mtime=0))})
    return {"source_digest": config["source_digest"], "protocol_digest": config["protocol_digest"],
            "all_jobs_complete": len(audits) == 12 and all(a["passed"] for a in audits),
            "logical_training_episode_slots": calls, "known_completed_episode_calls": known,
            "search_and_resume_elapsed_seconds": elapsed,
            "output_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
            "audit": audits, "observer_cost_probes": costs,
            "observer_probe_training_episodes_separate": sum(c["training_episodes"] for c in costs),
            "test_evaluations": 0, "selection_based_on_pilot_performance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "main", "confirmation"))
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("Use one to eight workers")
    root = ROOT / "studies/interaction/results_v02" / args.phase
    config = prepare(root, args.phase, gate=args.gate, pilot_report=args.pilot_report)
    started = time.perf_counter()
    if args.phase == "pilot":
        execute(root, config, args.workers, pause_after=2)
        execute(root, config, args.workers)
        write_json(root / "cost_report.json", pilot_report(root, config, time.perf_counter() - started))
    else:
        execute(root, config, args.workers)
        from .finalize import lock_all_candidates, evaluate_locked
        lock_all_candidates(root)
        evaluate_locked(root)


if __name__ == "__main__":
    main()
