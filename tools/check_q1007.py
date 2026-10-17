"""신규 예산 실행기와 기존 동결 구현의 검사 후 실행 게이트를 저장한다."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'studies/interaction'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        raise ValueError('새 작업 공간 게이트 파일 필요')
    sys.path.insert(0, str(STUDY))
    from evosim_q1007_v01.source import source_manifest, plan
    from evosim_q1007_v01.protocol import runtime
    from evosim_interaction_v02.storage import manifest_digest
    from evosim_interaction_v02.settings import digest_json
    before = source_manifest()
    commands = [(STUDY, [sys.executable, '-B', '-X', 'utf8', '-m', 'unittest', 'discover', '-s', name, '-v'])
                for name in ('evosim_q1007_v01/tests', 'evosim_q1006_v02/tests')]
    commands += [(ROOT, [sys.executable, '-B', '-X', 'utf8', 'tools/check_q1005_portable.py']),
                 (ROOT, [sys.executable, '-B', '-X', 'utf8', 'tools/check_research_layout.py', '--tests'])]
    records, logs = [], []
    for cwd, command in commands:
        start = time.perf_counter()
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, encoding='utf-8')
        log = result.stdout + result.stderr
        records.append({'cwd': cwd.relative_to(ROOT).as_posix(), 'command': command, 'returncode': result.returncode,
                        'tests': sum(map(int, re.findall(r'Ran (\d+) tests?', log))), 'seconds': time.perf_counter() - start})
        logs.append(log)
    value = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'passed': all(r['returncode'] == 0 for r in records) and before == source_manifest(),
             'full_gate': True, 'sources': before, 'source_digest': manifest_digest(before), 'plan_digest': digest_json(plan()),
             'runtime': runtime(), 'commands': records, 'tests': sum(r['tests'] for r in records),
             'new_tests': records[0]['tests'], 'policy_fixture_calls': '검사 전용; 공식 성능 집단과 분리. 전체 합계는 미집계',
             'independent_review': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    output.with_suffix('.log').write_text('\n'.join(logs), encoding='utf-8')
    print(json.dumps({k: value[k] for k in ('passed', 'source_digest', 'tests', 'new_tests')}, ensure_ascii=False))
    if not value['passed']:
        print('\n'.join(logs))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
