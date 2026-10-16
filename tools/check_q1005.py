"""Q1-005 구현과 동결 의존성·기존 검사를 함께 확인하는 실행 전 게이트."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/q1005_gate/gate.json")
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Use a new gate output path inside workspace")
    commands = [(ROOT / "studies/interaction", [sys.executable, "-B", "-X", "utf8", "-m", "unittest", "discover", "-s", "evosim_q1005_v01/tests", "-v"])]
    if not args.package_only:
        commands += [(ROOT, [sys.executable, "-B", "-X", "utf8", "tools/check_q1_core.py", "--output", "dist/q1005_gate/dependency.json"]),
                     (ROOT, [sys.executable, "-B", "-X", "utf8", "tools/check_research_layout.py", "--tests"]),
                     (ROOT, [sys.executable, "-B", "-X", "utf8", "-m", "unittest", "discover", "-s", "tools/tests", "-v"])]
    logs, entries = [], []
    for cwd, command in commands:
        started = time.perf_counter()
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
        log = result.stdout + result.stderr
        logs.append(log)
        match = re.search(r"Ran (\d+) tests?", log)
        entries.append({"cwd": cwd.relative_to(ROOT).as_posix(), "command": command,
                        "returncode": result.returncode, "tests": int(match[1]) if match else None,
                        "elapsed_seconds": time.perf_counter() - started})
    required = ["test_original_r0_kernel_exact_actions_and_metrics", "test_progress_upper_bound_is_attainable",
                "test_round_trip_does_not_repeat_progress", "test_scalar_batch_progress_and_reference_actions_match",
                "test_ordinal_s0_reproduction_matches_frozen_genotypes_rng_and_ancestry",
                "test_pause_resume_all_four_conditions_and_both_methods",
                "test_shared_random_genotypes_and_validation_winners_match_all_treatments",
                "test_mocked_test_pipeline_caches_three_roles_and_resumes_without_new_policy_test",
                "test_test_kernel_not_called_without_lock_or_for_another_genome_or_partial_test",
                "test_blind_first_tie_and_changed_validation_artifact_rejected",
                "test_unknown_interrupted_episode_work_keeps_upper_bound",
                "test_probe_single_step_accounting_has_separate_unknown_bound",
                "test_holm_three_contrasts_rejects_fifteen_positive_when_first_threshold_fails",
                "test_main_and_confirmation_summaries_share_only_one_random_control"]
    sys.path.insert(0, str(ROOT / "studies/interaction"))
    from evosim_q1005_v01.source import source_manifest, plan, DEPENDENCY_DIGEST
    from evosim_interaction_v02.storage import manifest_digest
    from evosim_interaction_v02.settings import digest_json
    sources = source_manifest()
    passed = all(e["returncode"] == 0 for e in entries) and all(name + " (" in logs[0] for name in required)
    previous_tests = None
    if not args.package_only:
        dependency = json.loads((ROOT / "dist/q1005_gate/dependency.json").read_text(encoding="utf-8"))
        legacy = json.loads((ROOT / "docs/management/verification.json").read_text(encoding="utf-8"))
        previous_tests = dependency["tests_run"] + sum(s["tests"] for s in legacy["studies"]) + entries[-1]["tests"]
        passed = passed and dependency["passed"] and legacy["passed"]
    record = {"checked_utc": datetime.now(timezone.utc).isoformat(), "passed": passed,
              "full_gate": not args.package_only, "source_digest": manifest_digest(sources), "sources": sources,
              "plan_digest": digest_json(plan()), "dependency_source_digest": DEPENDENCY_DIGEST,
              "new_test_count": entries[0]["tests"], "previous_test_count": previous_tests,
              "commands": entries, "required_test_names": required,
              "gate_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scientific_result": None, "independent_review": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    output.with_suffix(".log").write_text("\n".join(logs), encoding="utf-8", newline="\n")
    print(json.dumps({k: record[k] for k in ("passed", "full_gate", "source_digest", "new_test_count", "previous_test_count")}))
    if not passed:
        print("\n".join(logs))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
