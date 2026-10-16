"""완료한 Q1-005 원자료를 해시 목록과 함께 별도 ZIP으로 보관한다."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_catalog(args, manifest, payload, protocol, main_summary, confirmation, recovered=False):
    result = {"experiment_id": "Q1-005", "archive": args.output.relative_to(ROOT).as_posix(),
              "bytes": args.output.stat().st_size, "sha256": sha256(args.output), "files": len(manifest),
              "uncompressed_bytes": sum(row["bytes"] for row in manifest.values()),
              "manifest_sha256": hashlib.sha256(payload).hexdigest(),
              "source_digest": protocol["source_digest"], "protocol_digest": main_summary["protocol_digest"],
              "confirmation_condition": confirmation, "publication": "local archive; release URL pending",
              "catalog_recovered_from_existing_archive": recovered,
              "integrity_check": "input SHA-256 manifest and all ZIP CRC passed"}
    args.catalog.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-existing", action="store_true", help="보존한 기존 ZIP에서 목록만 복구")
    args = parser.parse_args()
    args.output, args.catalog = args.output.resolve(), args.catalog.resolve()
    if not args.output.is_relative_to(ROOT) or not args.catalog.is_relative_to(ROOT):
        raise ValueError("Archive and catalog must stay within the workspace")
    results = ROOT / "studies/interaction/results_q1005_v01"
    main_summary = json.loads((results / "main/summary.json").read_text(encoding="utf-8"))
    confirmation = main_summary["decision"]["confirmation_condition"]
    if confirmation and not (results / "confirmation/summary.json").exists():
        raise ValueError("Required confirmation is not yet complete")
    for phase in ("main", "confirmation") if confirmation else ("main",):
        audit = json.loads((results / phase / "test_audit.json").read_text(encoding="utf-8"))
        if not audit["passed"] or not all(row["passed"] for row in audit["jobs"]):
            raise ValueError("Production audit has not passed")
    archive = json.loads((ROOT / "docs/management/q1_005_source_archive.json").read_text(encoding="utf-8"))
    # 소스 보관 기록의 경로 필드는 키 이름과 무관하게 실제 ZIP을 확인한다.
    source_zips = list((ROOT / "dist").glob("q1005_source_*.zip"))
    if len(source_zips) != 1:
        raise ValueError("Expected one frozen source archive")
    if sha256(source_zips[0]) != archive["archive_sha256"]:
        raise ValueError("Frozen source archive digest mismatch")
    protocol = json.loads((results / "main/protocol.json").read_text(encoding="utf-8"))
    files = set(p for p in results.rglob("*") if p.is_file())
    for directory in ("dist/q1005_gate", "dist/q1005_cost_probe"):
        files.update(p for p in (ROOT / directory).rglob("*") if p.is_file())
    files.update(source_zips)
    files.update(p for p in (ROOT / "dist").glob("q1005_*.log") if p.is_file())
    files.update(p for p in (ROOT / "docs/management").glob("q1_005_*.json") if p.is_file())
    files.update(p for p in (ROOT / "docs/management").glob("q1_005_*.log") if p.is_file())
    files.add(ROOT / "docs/protocols/q1_005_v01_plan.json")
    if args.catalog.exists():
        raise FileExistsError("Catalog already exists; never overwrite")
    if args.catalog_existing:
        with zipfile.ZipFile(args.output) as check:
            payload = check.read("MANIFEST.json")
            manifest = json.loads(payload)["files"]
            names = check.namelist()
            expected = {p.relative_to(ROOT).as_posix() for p in files}
            if set(manifest) != expected or set(names) != expected | {"MANIFEST.json"} or len(set(names)) != len(names):
                raise ValueError("Existing archive membership mismatch")
            if check.testzip():
                raise ValueError("Existing archive CRC check failed")
        write_catalog(args, manifest, payload, protocol, main_summary, confirmation, recovered=True)
        return
    if args.output.exists():
        raise FileExistsError("Archive and catalog are immutable; select new paths")
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError("Previous incomplete archive retained")
    manifest = {}
    with zipfile.ZipFile(partial, "x", allowZip64=True) as output:
        for index, path in enumerate(sorted(files)):
            relative = path.relative_to(ROOT).as_posix()
            before = path.stat()
            entry = zipfile.ZipInfo(relative)
            entry.compress_type = zipfile.ZIP_STORED if path.suffix in (".gz", ".zip") else zipfile.ZIP_DEFLATED
            digest, size = hashlib.sha256(), 0
            with path.open("rb") as source, output.open(entry, "w", force_zip64=True) as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                    target.write(block)
            if (before.st_size, before.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns):
                raise ValueError(f"File changed during packaging: {relative}")
            manifest[relative] = {"bytes": size, "sha256": digest.hexdigest()}
            if (index + 1) % 2000 == 0:
                print(f"ARCHIVE {index + 1}/{len(files)} files", flush=True)
        payload = json.dumps({"files": manifest}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        output.writestr("MANIFEST.json", payload, compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(partial) as check:
        bad = check.testzip()
        if bad:
            raise ValueError(f"Archive CRC check failed: {bad}")
    partial.rename(args.output)
    write_catalog(args, manifest, payload, protocol, main_summary, confirmation)


if __name__ == "__main__":
    main()
