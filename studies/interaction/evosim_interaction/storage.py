"""Atomic, inspectable records and source manifests."""

import gzip
import hashlib
import json
import os
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if path.suffix == ".gz":
        data = gzip.compress(data, compresslevel=3, mtime=0)
    temporary.write_bytes(data)
    os.replace(temporary, path)


def read_json(path):
    path = Path(path)
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return json.loads(data)


def source_manifest():
    root = Path(__file__).parent
    return {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py")) if "results" not in path.parts}


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def as_tuple(value):
    return tuple(as_tuple(item) for item in value) if isinstance(value, list) else value
