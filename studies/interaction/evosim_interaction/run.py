"""Stage-specific entry points; test data cannot be requested by pilot/search."""

import argparse
import concurrent.futures
import platform
from pathlib import Path

import numpy as np

from .datasets import make_split
from .network import SETTINGS
from .search import run_search
from .storage import manifest_digest, read_json, source_manifest, write_json


def prepare(root, pilot=False, seeds=None, rounds=None):
    root = Path(root)
    manifest = source_manifest()
    split = make_split()
    config = {
        "version": "q1-v0.1", "pilot": pilot,
        "seeds": seeds or ([901, 902] if pilot else list(range(201, 211))),
        "rounds": rounds or (10 if pilot else 100), "population_size": 20,
        "training_episodes_per_round": 24, "evaluation_repeats": 3,
        "methods": ["evolution", "random"], "settings": SETTINGS,
        "split_digest": split.digest, "split_seed": 71001,
        "source_digest": manifest_digest(manifest), "sources": manifest,
        "python": platform.python_version(), "numpy": np.__version__,
        "decision": {"minimum_difference": 0.10, "positive_seeds": 8,
                     "probe_threshold": 0.80, "probe_passing_seeds": 8,
                     "bootstrap_samples": 10000, "bootstrap_seed": 71004,
                     "ci_percentiles": [2.5, 97.5]},
    }
    path = root / "protocol.json"
    if path.exists():
        if read_json(path) != config:
            raise ValueError("Protocol or source changed; preserve the old run and choose a new output path")
    else:
        write_json(root / "datasets.json", split.manifest())
        write_json(path, config)
    return config


def execute_searches(root, config, workers):
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_search, root, seed, method,
                                   config["rounds"], config["population_size"], config["pilot"]): (seed, method)
                   for seed in config["seeds"] for method in config["methods"]}
        for future in concurrent.futures.as_completed(futures):
            seed, method = futures[future]
            try:
                result = future.result()
                print(f"DONE {method} {seed}: {result['elapsed_seconds']:.1f}s", flush=True)
            except Exception as exc:
                failures.append({"seed": seed, "method": method, "error": repr(exc)})
                write_json(Path(root) / "failures.json", failures)
                print(f"FAILED {method} {seed}: {exc!r}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} search jobs failed; records retained")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pilot", "search"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config = prepare(args.output, pilot=args.stage == "pilot")
    execute_searches(args.output, config, args.workers)


if __name__ == "__main__":
    main()
