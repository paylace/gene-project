"""Q1-009 ZIP을 검증하고 기존 파일을 덮어쓰지 않으며 복원한다."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def stream_digest(handle):
    value = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "evidence/q1_009_v01.json").read_text(encoding="utf-8"))
    if args.archive.name != Path(catalog["archive"]).name or args.archive.stat().st_size != catalog["bytes"]:
        raise ValueError("Unregistered archive name or size")
    with args.archive.open("rb") as handle:
        if stream_digest(handle) != catalog["sha256"]:
            raise ValueError("Archive SHA-256 mismatch")
    missing = []
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or len(names) != catalog["files"] + 1:
            raise ValueError("Duplicate or incorrect archive member count")
        payload = archive.read("MANIFEST.json")
        if hashlib.sha256(payload).hexdigest() != catalog["manifest_sha256"]:
            raise ValueError("Manifest SHA-256 mismatch")
        manifest = json.loads(payload)["files"]
        if set(names) != set(manifest) | {"MANIFEST.json"}:
            raise ValueError("Manifest membership mismatch")
        for index, (name, expected) in enumerate(manifest.items()):
            target = (ROOT / name).resolve()
            if "\\" in name or ".." in PurePosixPath(name).parts or not target.is_relative_to(ROOT):
                raise ValueError(f"Unsafe archive path: {name}")
            if archive.getinfo(name).file_size != expected["bytes"]:
                raise ValueError(f"Member size mismatch: {name}")
            with archive.open(name) as handle:
                if stream_digest(handle) != expected["sha256"]:
                    raise ValueError(f"Member SHA-256 mismatch: {name}")
            if target.exists():
                with target.open("rb") as handle:
                    if stream_digest(handle) != expected["sha256"]:
                        raise ValueError(f"Refuse to overwrite changed file: {name}")
            else:
                missing.append((name, target))
            if (index + 1) % 2000 == 0:
                print(f"VERIFY {index + 1}/{len(manifest)} files", flush=True)
        if not args.verify_only:
            for name, target in missing:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
    print(f"Verified {catalog['files']} files; missing={len(missing)}; restored={0 if args.verify_only else len(missing)}")


if __name__ == "__main__":
    main()
