"""Check active document links, terminal attribution, locked code and moved tests."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MANAGEMENT = ROOT / "docs" / "management"


def save(name, value):
    (MANAGEMENT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate(require_data=False):
    catalog_path = ROOT / 'evidence/catalog.json'
    external = json.loads(catalog_path.read_text(encoding='utf-8'))['files'] if catalog_path.exists() else {}
    deferred = []
    active = [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "연구_목표와_진행계획.md",
              ROOT / "studies/interaction/README.md", ROOT / "studies/interaction/연구_목표와_진행계획.md"]
    active += sorted((ROOT / "docs").rglob("*.md"))
    errors, links = [], 0
    for path in active:
        content = path.read_text(encoding="utf-8")
        # Fenced examples are explanatory code, not live document links.
        content = re.sub(r"```.*?```", "", content, flags=re.S)
        for target in re.findall(r"\[[^\]]*\]\((<[^>]+>|[^)]+)\)", content):
            target = target.strip().strip("<>")
            if target.startswith(("https:", "http:", "mailto:", "#", "codex:", "app:")):
                continue
            target = unquote(target.split("#", 1)[0])
            destination = (path.parent / target).resolve()
            relative = destination.relative_to(ROOT).as_posix() if destination.is_relative_to(ROOT) else None
            known_external = relative in external or (relative and any(n.startswith(relative + '/') for n in external))
            if not destination.exists() and known_external and not require_data:
                deferred.append(relative)
            elif not destination.is_relative_to(ROOT) or not destination.exists():
                errors.append(f"Broken/outside link in {path.relative_to(ROOT)}: {target}")
            links += 1
    reports = sorted(path for path in (ROOT / "docs/reports").glob("*.md") if path.name != "README.md")
    identifiers = set()
    for path in reports:
        text = path.read_text(encoding="utf-8")
        identifier_match = re.fullmatch(r"([A-Z]+[0-9]*-\d{3,})_.+\.md", path.name)
        if identifier_match is None:
            errors.append(f"Invalid report filename: {path.name}")
        elif identifier_match.group(1) in identifiers:
            errors.append(f"Duplicate report identifier: {path.name}")
        else:
            identifiers.add(identifier_match.group(1))
        heading = "## 수행·작성·검토 기록"
        if text.count(heading) != 1:
            errors.append(f"Missing/duplicate terminal attribution: {path.name}")
            continue
        footer = text.split(heading, 1)[1]
        if re.search(r"^#{1,6} ", footer, flags=re.M):
            errors.append(f"Attribution must be the final section: {path.name}")
        for field in ("연구 방향·실험 요청자:", "실험 수행자:", "실험 수행일:", "보고서 작성자:",
                      "모델·버전:", "독립 검토 상태:"):
            if field not in footer:
                errors.append(f"Missing {field} in {path.name}")
        if not any(field in footer for field in ("문서 정리일:", "문서 작성일:", "문서 작성·정리일:", "문서 작성일 또는 정리일:")):
            errors.append(f"Missing document date in {path.name}")
        if re.search(r"<[^>]+>", text):
            errors.append(f"Unfilled placeholder: {path.name}")
    if not reports:
        errors.append("No official research reports found")
    locked = []
    for project, relative, field, source_root in (
        ("working_memory", "research_results/protocol.json", "source_sha256", ROOT / "studies/working_memory"),
        ("interaction", "evosim_interaction/results_v01/protocol.json", "sources", ROOT / "studies/interaction/evosim_interaction"),
    ):
        protocol_path = ROOT / "studies" / project / relative
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        for name, digest in protocol[field].items():
            path = source_root / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            if actual != digest:
                errors.append(f"Locked source mismatch: {project}/{name}")
            locked.append(f"{project}/{name}")
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "passed": not errors,
              "active_markdown_files": len(active), "local_links": links,
              "official_reports": len(reports), "locked_source_files": len(locked), "external_data_links_deferred": sorted(set(deferred)), "errors": errors}
    save("layout_check.json", result)
    print(json.dumps(result, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


def test_moved_projects():
    temporary = (ROOT / "tools" / ".verification_tmp").resolve()
    if not temporary.is_relative_to(ROOT):
        raise ValueError("Temporary test directory must be in the workspace")
    temporary.mkdir(exist_ok=True)
    entries, outputs = [], []
    for label, project, tests in (
        ("working_memory", ROOT / "studies/working_memory", "tests"),
        ("interaction", ROOT / "studies/interaction", "evosim_interaction/tests"),
    ):
        code = (
            "import tempfile, unittest; "
            f"tempfile.tempdir = {str(temporary)!r}; "
            f"suite = unittest.defaultTestLoader.discover({tests!r}); "
            "result = unittest.TextTestRunner(verbosity=2).run(suite); "
            "raise SystemExit(0 if result.wasSuccessful() else 1)"
        )
        completed = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", code], cwd=project,
                                   text=True, encoding="utf-8", errors="replace", capture_output=True)
        output = completed.stdout + completed.stderr
        match = re.search(r"Ran (\d+) tests?", output)
        entry = {"study": label, "cwd": project.relative_to(ROOT).as_posix(),
                 "returncode": completed.returncode, "tests": int(match.group(1)) if match else None,
                 "temporary_directory": temporary.relative_to(ROOT).as_posix()}
        entries.append(entry)
        outputs.append(f"=== {label} ===\n{output}")
        print(json.dumps(entry), flush=True)
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "purpose": "relocation verification, no experiments rerun",
              "python": sys.version, "passed": all(item["returncode"] == 0 for item in entries), "studies": entries}
    save("verification.json", result)
    (MANAGEMENT / "verification_console.log").write_text("\n\n".join(outputs), encoding="utf-8")
    if not result["passed"]:
        print("\n\n".join(outputs))
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--require-data", action="store_true")
    args = parser.parse_args()
    validate(args.require_data)
    if args.tests:
        test_moved_projects()
