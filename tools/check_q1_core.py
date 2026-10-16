"""Q1-004 구현 검사와 코드 해시 기록. 공식 예비·본 탐색은 실행하지 않는다."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/interaction"
PACKAGE = STUDY / "evosim_interaction_v02"


def hashes(paths):
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def check(output):
    if not output.resolve().is_relative_to(ROOT):
        raise ValueError("검사 기록은 저장소 내부에만 씁니다")
    baseline = json.loads((STUDY / "evosim_interaction/results_v01/protocol.json").read_text(encoding="utf-8"))
    locked = [STUDY / "evosim_interaction" / name.replace("\\", "/") for name in baseline["sources"]]
    sources = sorted(PACKAGE.rglob("*.py")) + [Path(__file__).resolve()]
    before = hashes(sources)
    baseline_before = hashes(locked)
    mismatches = [name for name, digest in baseline["sources"].items()
                  if baseline_before[(STUDY / "evosim_interaction" / name.replace("\\", "/")).relative_to(ROOT).as_posix()] != digest]
    command = [sys.executable, "-B", "-X", "utf8", "-m", "unittest", "discover", "-s", "evosim_interaction_v02/tests", "-v"]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=STUDY, capture_output=True, text=True, encoding="utf-8")
    console = completed.stdout + completed.stderr
    after, baseline_after = hashes(sources), hashes(locked)
    count = re.search(r"Ran (\d+) tests?", console)
    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Q1-004 implementation verification, not performance experiment",
        "passed": completed.returncode == 0 and count is not None and not mismatches
                  and before == after and baseline_before == baseline_after,
        "tests_run": int(count.group(1)) if count else 0,
        "returncode": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version,
        "cwd": STUDY.relative_to(ROOT).as_posix(),
        "command": ["python"] + command[1:],
        "implementation_sources": before,
        "frozen_reference_files": len(locked), "frozen_reference_mismatches": mismatches,
        "source_unchanged_during_tests": before == after and baseline_before == baseline_after,
        "new_pilot_searches": 0, "new_main_searches": 0,
        "evaluation_data_used": "finite fixtures, train-only verification searches, coordinate-policy foundation; no evolved benchmark test scores",
        "independent_review": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    output.with_suffix(".log").write_text(console, encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        print(console)
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/q1_core_verification/checks.json")
    args = parser.parse_args()
    check(args.output)
