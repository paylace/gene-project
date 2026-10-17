"""평가→쓰기→복원 검증의 통합 비용. 기술 호출의 사전 상한을 유지한다."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'studies/interaction'
sys.path.insert(0, str(STUDY))
from evosim_runtime_v01.storage import canonical, digest
from evosim_runtime_v01.accounting import COUNTS
from evosim_q1007_v01.resources import peak_memory_bytes
from experiment_runtime_v01 import ExperimentRuntime


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    from evosim_q1007_v02.provenance import source_manifest
    return {**source_manifest(), **{p.relative_to(ROOT).as_posix(): sha(p) for p in
            (STUDY / 'evosim_runtime_v02').rglob('*.py')},
            'facade': sha(STUDY / 'experiment_runtime_v01.py'), 'tool': sha(Path(__file__))}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value))


def worker(out, index):
    from evosim_interaction_v02.datasets import make_split
    from evosim_interaction_v02.rollout import derive_seed
    from evosim_q1007_v01.network import initial_genome
    plan = json.loads((out / 'plan.json').read_bytes())
    if sources() != plan['sources']:
        raise ValueError('측정 중 소스 변경')
    task = plan['schedule'][index]
    size, count, repeats = task['case']
    seed = 3101 if size == 20 else 3102
    rng = random.Random(derive_seed('q1007:v01:initial', seed))
    genomes = [initial_genome(rng) for _ in range(size)]
    episodes = make_split().train[:count]
    purposes = [(f'train:0:q1007:v01:slot{i}' if count == 24 else f'cost:q1011:slot{i}') for i in range(size)]
    path = out / 'records' / (task['name'] + '.json.gz')
    save(out / 'attempts' / (task['name'] + '.json'), dict(task=task, status='started', maximum_calls=size * count * repeats))
    payload, timings = ExperimentRuntime(task['profile']).evaluate_and_store(path, genomes, episodes,
        seed, purposes, training=count == 24, repeats=repeats)
    result = dict(task=task, timings=timings, bytes=path.stat().st_size, file_sha256=sha(path),
                  result_sha256=digest(payload), peak_memory_bytes=peak_memory_bytes(), accounting=dict(COUNTS))
    save(out / 'measurements' / (task['name'] + '.json'), result)
    print(json.dumps(dict(completed=task['name'], total_seconds=timings['total_seconds'])), flush=True)


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    schedule = []
    for size, episodes, repeats, runs in ((20, 24, 1, 3), (40, 24, 1, 3), (20, 252, 3, 2)):
        for repeat in range(runs):
            schedule.extend(dict(name=f'p{size}_e{episodes}_{profile}_{repeat}', profile=profile,
                                 case=[size, episodes, repeats], repeat=repeat) for profile in ('reference', 'balanced'))
    random.Random(71103).shuffle(schedule)
    assert sum(t['case'][0] * t['case'][1] * t['case'][2] for t in schedule) == 69120
    plan = dict(experiment_id='Q1-011', sources=sources(), schedule=schedule,
        expected_policy_episode_calls=69120, overall_policy_episode_calls_cap=380800, independent_review=False)
    save(out / 'plan.json', plan)
    for index in range(len(schedule)):
        subprocess.run([sys.executable, '-B', '-u', '-X', 'utf8', str(Path(__file__).resolve()),
                        '--output', str(out), '--worker', str(index)], check=True)
    measurements = [json.loads(p.read_bytes()) for p in sorted((out / 'measurements').glob('*.json'))]
    groups = {}
    for case in ([20, 24, 1], [40, 24, 1], [20, 252, 3]):
        group = [m for m in measurements if m['task']['case'] == case]
        if len({m['result_sha256'] for m in group}) != 1:
            raise ValueError('프로필에 따라 원본 결과가 다름')
        row = dict(case=case, all_results_identical=True)
        for profile in ('reference', 'balanced'):
            arm = [r for r in group if r['task']['profile'] == profile]
            row[profile] = dict(runs=len(arm), seconds_runs=[r['timings']['total_seconds'] for r in arm],
                median_seconds=statistics.median(r['timings']['total_seconds'] for r in arm),
                bytes=arm[0]['bytes'], peak_memory_bytes_max=max(r['peak_memory_bytes'] for r in arm),
                component_medians={k: statistics.median(r['timings'][k] for r in arm) for k in arm[0]['timings']})
        row['speedup'] = row['reference']['median_seconds'] / row['balanced']['median_seconds']
        groups[f'p{case[0]}_e{case[1]}'] = row
    calls = sum(m['accounting']['reference_episodes'] + m['accounting']['optimized_episodes'] for m in measurements)
    if calls != 69120 or any(m['accounting']['unknown_failed_episodes_upper'] for m in measurements):
        raise ValueError('통합 호출 원장 불일치')
    assert sources() == plan['sources']
    save(out / 'summary.json', dict(passed=True, groups=groups, policy_episode_calls=calls,
         main_or_heldout_partition_calls=0, source_digest=digest(plan['sources']), plan_sha256=sha(out / 'plan.json'), independent_review=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=STUDY / 'optimization_q1011_pipeline_v01')
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError('작업 공간 출력만 허용')
    worker(out, args.worker) if args.worker is not None else main(out)
