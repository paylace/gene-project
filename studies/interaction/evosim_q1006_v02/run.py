"""예비→본→조건부 확인의 공식 실행 진입점."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

from evosim_interaction_v02.storage import read_json, write_json, write_once
from .protocol import prepare, RESULTS
from .source import plan
from .search import run_search, directory, attempt_accounting
from .audit import audit_job


def execute(root, config, workers, pause_after=None):
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_search, root, **job, pause_after=pause_after): job for job in config["jobs"]}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
                print(f"DONE {job['condition']}/{job['method']}/{job['seed']}", flush=True)
            except Exception as error:
                errors.append({"job": job, "error": repr(error)})
                write_json(Path(root) / "failures.json", errors)
    if errors:
        raise RuntimeError("Search errors retained; inspect before continuing")


def pilot_report(root, config, elapsed):
    audits, slots, known, unknown = [], 0, 0, 0
    for job in config["jobs"]:
        audits.append(audit_job(root, **job))
        path = directory(root, **job)
        slots += read_json(path / "result.json")["logical_episode_slots"]
        accounting = attempt_accounting(path)
        known += accounting["known_completed_episode_calls"]
        unknown += accounting["interrupted_attempts"]
    if unknown:
        raise ValueError("Pilot contains unknown interrupted work; inspect accounting")
    return {"phase": "pilot", "source_digest": config["source_digest"], "plan_digest": config["plan_digest"],
            "protocol_digest": config["protocol_digest"], "all_jobs_complete": len(audits) == 8 and all(a["passed"] for a in audits),
            "logical_training_episode_slots": slots, "known_completed_episode_calls": known,
            "unknown_interrupted_attempts": unknown, "search_and_resume_seconds": elapsed,
            "output_bytes": sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file()),
            "audits": audits, "validation_or_test_evaluations": 0, "performance_based_selection": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "main", "confirmation"))
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("Use one to eight worker processes")
    root = RESULTS / args.phase
    if (root / "failures.json").exists():
        raise ValueError("Prior failures require explicit inspection; preserved")
    config = prepare(root, args.phase, gate=args.gate, pilot_report=args.pilot_report)
    started = time.perf_counter()
    if args.phase == "pilot":
        execute(root, config, args.workers, pause_after=2)
        execute(root, config, args.workers)
        report=pilot_report(root, config, time.perf_counter() - started)
        write_once(root / "cost_report.json", report)
        if not report["all_jobs_complete"] or report["known_completed_episode_calls"]!=19200:
            raise ValueError("Pilot composition/budget differs; evidence preserved")
    else:
        execute(root, config, args.workers)
        from .finalize import lock_all, evaluate_locked
        lock_all(root)
        evaluate_locked(root)


if __name__ == "__main__":
    main()
