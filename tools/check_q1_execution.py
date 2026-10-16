"""Q1-004 예비 진입 전의 구현·기존 검사·계획 검사 게이트를 기록한다."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Gate output must be in workspace")
    commands = [
        [sys.executable, "-B", "-X", "utf8", "tools/check_q1_core.py", "--output", "dist/q1_execution_gate/core.json"],
        [sys.executable, "-B", "-X", "utf8", "tools/check_research_layout.py", "--tests"],
        [sys.executable, "-B", "-X", "utf8", "-m", "unittest", "discover", "-s", "tools/tests", "-v"],
    ]
    outputs, codes = [], []
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        outputs.append(result.stdout + result.stderr)
        codes.append(result.returncode)
    core = json.loads((ROOT / "dist/q1_execution_gate/core.json").read_text(encoding="utf-8"))
    console = (ROOT / "dist/q1_execution_gate/core.log").read_text(encoding="utf-8")
    required = ["test_observer_mirror_does_not_change_actions_rewards_or_genome",
                "test_pause_resume_for_each_condition_and_method", "test_orphan_record_replayed_without_overwriting",
                "test_complete_genetic_audit_all_conditions_and_methods", "test_locked_candidates_and_blind_tampering_are_rejected",
                "test_primary_selection_priority_and_mean_gate", "test_probe_and_all_1512_coordinate_episodes",
                "test_summary_main_and_confirmation_use_registered_rules"]
    sys.path.insert(0, str(ROOT / "studies/interaction"))
    from evosim_interaction_v02.storage import manifest_digest, source_manifest
    sources = source_manifest()
    layout = json.loads((ROOT / "docs/management/layout_check.json").read_text(encoding="utf-8"))
    legacy = json.loads((ROOT / "docs/management/verification.json").read_text(encoding="utf-8"))
    passed = all(code == 0 for code in codes) and core["passed"] and layout["passed"] and legacy["passed"]
    passed = passed and all(name + " (" in console and name in console for name in required)
    record = {"checked_utc": datetime.now(timezone.utc).isoformat(), "passed": passed,
              "gate": "Q1-004 W3-W6 implementation checks before pilot",
              "source_digest": manifest_digest(sources), "sources": sources,
              "core_test_count": core["tests_run"], "legacy_test_count": sum(s["tests"] for s in legacy["studies"]),
              "plan_test_count": 12, "core": core, "locked_source_files_verified": layout["locked_source_files"],
              "required_test_names": required, "returncodes": codes,
              "gate_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scientific_conclusion": None, "independent_review": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    output.with_suffix(".log").write_text("\n".join(outputs) + "\n" + console, encoding="utf-8", newline="\n")
    print(json.dumps({key: record[key] for key in ("passed", "source_digest", "core_test_count", "legacy_test_count", "plan_test_count")}))
    if not passed:
        print("\n".join(outputs))
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/q1_execution_gate/gate.json")
    main(parser.parse_args().output)
