"""신규 기반 검사와 실제 기술 호출을 별도 실행 폴더에 기록한다."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'studies/interaction'))
from evosim_runtime_v01.accounting import COUNTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('작업 공간의 새 검사 출력 폴더 필요')
    output.mkdir(parents=True)
    files = [p for folder in ('evosim_runtime_v01', 'evosim_q1007_v02')
             for p in (ROOT / 'studies/interaction' / folder).rglob('*.py')]
    before = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for name in before:
        target = output / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (output / 'sources.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'studies/interaction/evosim_runtime_v01/tests'))
    with (output / 'console.log').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    stable = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in before.items())
    report = dict(passed=result.wasSuccessful() and stable, tests=result.testsRun,
                  sources_stable=stable, policy_accounting=dict(COUNTS), independent_review=False)
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    if not report['passed']:
        print((output / 'console.log').read_text(encoding='utf-8'))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
