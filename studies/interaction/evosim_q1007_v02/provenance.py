"""기술 어댑터·공통 기반과 모든 선행 의존 소스를 함께 식별한다."""
import hashlib
from pathlib import Path
from evosim_q1007_v01.source import source_manifest as inherited

ROOT = Path(__file__).resolve().parents[3]


def source_manifest():
    result = {'inherited/' + k: v for k, v in inherited().items()}
    for package in ('evosim_runtime_v01', 'evosim_q1007_v02'):
        for path in sorted((ROOT / 'studies/interaction' / package).rglob('*.py')):
            result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
