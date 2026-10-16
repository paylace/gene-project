"""새 코드와 기존 동결 의존성을 함께 확인한다."""

import hashlib
from pathlib import Path

from evosim_interaction_v02.storage import read_json, manifest_digest
from evosim_interaction_v02 import storage as dependency_storage

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "docs/protocols/q1_005_v01_plan.json"
DEPENDENCY_DIGEST = "032467c82315c7e79f037de24b0636b0c65aa953d5bfbedeb954539fab04a984"


def source_manifest():
    dependency = dependency_storage.source_manifest()
    if manifest_digest(dependency) != DEPENDENCY_DIGEST:
        raise ValueError("Frozen Q1-004 dependency changed")
    here = Path(__file__).parent
    own = {p.relative_to(here).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(here.rglob("*.py"))}
    return {**{"dependency/" + k: v for k, v in dependency.items()},
            **{"q1005/" + k: v for k, v in own.items()}}


def plan():
    return read_json(PLAN_PATH)
