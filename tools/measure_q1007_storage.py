"""Q1-007 완료 예비 자료의 손실 없는 저장 비교. 정책 평가를 호출하지 않는다."""
import argparse
import base64
import copy
import gzip
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'studies/interaction'
sys.path.insert(0, str(STUDY))
from evosim_interaction_v02.environment import EpisodeSpec, LineCollectEnv
from evosim_q1007_v01.resources import peak_memory_bytes

SOURCE = STUDY / 'results_q1007_v01/pilot'
DEFAULT_OUTPUT = STUDY / 'storage_q1007_v01'
MODES = ('gzip3', 'gzip9', 'shared', 'reconstruct', 'dictionary', 'packed')
FIELDS = {'episode_id', 'repeat', 'actions', 'rewards_cents', 'collected',
          'success', 'walls', 'failed_collect', 'optimal_steps', 'final_position'}
RECORD_KEYS = {'records', 'training_records', 'validation_records'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value))


def derive_record(episode_id, actions, training):
    if not isinstance(actions, str) or not 1 <= len(actions) <= 24 or set(actions) - set('012'):
        raise ValueError('Invalid actions')
    start, goals = episode_id.split(':')
    episode = EpisodeSpec(int(start), tuple(map(int, goals.split(','))))
    if episode.episode_id != episode_id:
        raise ValueError('Noncanonical episode id')
    env = LineCollectEnv()
    env.reset(episode)
    rewards, progress = [], []
    walls = failed = 0
    minimum = abs(env.state.position - env.state.goal)
    for action in actions:
        before = env.state
        step = env.step(int(action))
        after = env.state
        distance = abs(after.position - before.goal)
        progress.append(max(0, minimum - distance) if action != '2' else 0)
        minimum = min(minimum, distance)
        if step.collected and not step.terminated:
            minimum = abs(after.position - after.goal)
        rewards.append(round(step.reward * 100))
        walls += int(step.wall_collision)
        failed += int(step.failed_collect)
    if not env.state.terminated:
        raise ValueError('Incomplete trajectory')
    row = dict(episode_id=episode_id, actions=actions, rewards_cents=rewards,
               collected=env.state.collected, success=env.state.collected == 3,
               walls=walls, failed_collect=failed, optimal_steps=episode.optimal_steps,
               final_position=env.state.position)
    if training:
        row['progress_units'] = progress
    return row


def pack(actions):
    raw = bytearray((len(actions) + 3) // 4)
    for index, action in enumerate(actions):
        raw[index // 4] |= int(action) << (2 * (index % 4))
    return [len(actions), base64.b64encode(raw).decode('ascii')]


def unpack(item):
    length, encoded = item
    if type(length) is not int or not 1 <= length <= 24:
        raise ValueError('Invalid packed length')
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != (length + 3) // 4:
        raise ValueError('Invalid packed bytes')
    actions = ''.join(str((raw[i // 4] >> (2 * (i % 4))) & 3) for i in range(length))
    if '3' in actions or pack(actions) != item:
        raise ValueError('Invalid action or padding')
    return actions


def encode(value, mode):
    tables = {'genomes': [], 'orders': [], 'actions': []}
    indexes = {key: {} for key in tables}

    def intern(kind, item):
        key = canonical(item)
        if key not in indexes[kind]:
            indexes[kind][key] = len(tables[kind])
            tables[kind].append(item)
        return indexes[kind][key]

    def visit(item):
        if isinstance(item, list):
            return [visit(v) for v in item]
        if not isinstance(item, dict):
            return item
        result = {}
        for key, data in item.items():
            if key == 'genome':
                result[key] = intern('genomes', data)
            elif key in RECORD_KEYS:
                if not data:
                    raise ValueError('Empty record block')
                fields = set(data[0])
                training = 'progress_units' in fields
                if fields != FIELDS | ({'progress_units'} if training else set()):
                    raise ValueError('Unsupported record schema')
                if any(set(row) != fields or type(row['repeat']) is not int or row['repeat'] < 0 for row in data):
                    raise ValueError('Inconsistent schema/repeat')
                order = [[row['episode_id'], row['repeat']] for row in data]
                block = {'order': intern('orders', order), 'training': training}
                if mode == 'shared':
                    block['columns'] = {k: [r[k] for r in data] for k in sorted(fields - {'episode_id', 'repeat'})}
                elif mode == 'reconstruct':
                    block['actions'] = [r['actions'] for r in data]
                else:
                    block['action_refs'] = [intern('actions', r['actions']) for r in data]
                result[key] = block
            else:
                result[key] = visit(data)
        return result

    payload = visit(value)
    if mode == 'packed':
        tables['actions'] = [pack(s) for s in tables['actions']]
    body = {'schema': 'q1007-storage-probe-1', 'mode': mode, 'tables': tables,
            'payload': payload, 'original_canonical_sha256': sha(canonical(value))}
    return {**body, 'content_sha256': sha(canonical(body))}


def decode(envelope):
    body = {k: v for k, v in envelope.items() if k != 'content_sha256'}
    if sha(canonical(body)) != envelope['content_sha256']:
        raise ValueError('Encoded content checksum mismatch')
    if body['schema'] != 'q1007-storage-probe-1' or body['mode'] not in MODES[2:]:
        raise ValueError('Unknown schema/mode')
    mode, tables = body['mode'], body['tables']
    actions_table = [unpack(v) for v in tables['actions']] if mode == 'packed' else tables['actions']
    memo = {}

    def ref(values, index):
        if type(index) is not int or not 0 <= index < len(values):
            raise ValueError('Invalid reference')
        return values[index]

    def visit(item):
        if isinstance(item, list):
            return [visit(v) for v in item]
        if not isinstance(item, dict):
            return item
        result = {}
        for key, data in item.items():
            if key == 'genome':
                result[key] = copy.deepcopy(ref(tables['genomes'], data))
            elif key in RECORD_KEYS:
                order = ref(tables['orders'], data['order'])
                records = []
                if mode == 'shared':
                    columns = data['columns']
                    if any(len(v) != len(order) for v in columns.values()):
                        raise ValueError('Column length differs')
                    for index, (ep, rep) in enumerate(order):
                        records.append(dict(episode_id=ep, repeat=rep, **{k: copy.deepcopy(v[index]) for k, v in columns.items()}))
                else:
                    actions = data['actions'] if mode == 'reconstruct' else [ref(actions_table, i) for i in data['action_refs']]
                    if len(actions) != len(order):
                        raise ValueError('Action/order length differs')
                    for (ep, rep), action in zip(order, actions):
                        memo_key = (ep, action, data['training'])
                        if memo_key not in memo:
                            memo[memo_key] = derive_record(*memo_key)
                        records.append(dict(copy.deepcopy(memo[memo_key]), repeat=rep))
                result[key] = records
            else:
                result[key] = visit(data)
        return result

    restored = visit(body['payload'])
    if sha(canonical(restored)) != body['original_canonical_sha256']:
        raise ValueError('Restored original checksum mismatch')
    return restored


def selftest():
    # 명시한 고정 행동의 기록을 구성한다. 네트워크 호출은 없다.
    row = dict(derive_record('0:1,2,3', '121212', True), repeat=0)
    sample = {'candidates': [{'genome': {'weights': [1.25]}, 'training_records': [row], 'ties': 7}]}
    passed = []
    for mode in MODES[2:]:
        packed = encode(sample, mode)
        assert canonical(decode(packed)) == canonical(sample)
        passed.append(mode + ':roundtrip')
        bad = copy.deepcopy(packed)
        bad['payload']['candidates'][0]['ties'] = 8
        try:
            decode(bad)
        except ValueError:
            passed.append(mode + ':content_tamper_rejected')
        else:
            raise AssertionError('Tamper accepted')
        # Re-sign the outer envelope: original full-record checksum must still reject.
        bad['content_sha256'] = sha(canonical({k: v for k, v in bad.items() if k != 'content_sha256'}))
        try:
            decode(bad)
        except ValueError:
            passed.append(mode + ':original_tamper_rejected')
        else:
            raise AssertionError('Original tamper accepted')
    for length in range(1, 25):
        for offset in range(3):
            actions = ''.join(str((i + offset) % 3) for i in range(length))
            assert unpack(pack(actions)) == actions
    passed.append('packed:72_length_patterns')
    for invalid in ([1, 'Aw=='], [1, 'BA=='], [24, 'AA==']):
        try:
            unpack(invalid)
        except ValueError:
            continue
        raise AssertionError('Invalid packed data accepted')
    passed.append('packed:invalid_action_padding_length')
    return passed


def worker(output, mode, repeat):
    plan = json.loads((output / 'plan.json').read_bytes())
    if sha(Path(__file__).read_bytes()) != plan['tool_sha256']:
        raise ValueError('Tool changed after plan')
    totals = dict(source_read_seconds=0., encode_seconds=0., restore_seconds=0.,
                  original_stored_bytes=0, original_json_bytes=0, stored_bytes=0,
                  decoded_bytes=0, files=0, episode_records=0)
    details = []
    for relative in plan['trajectory_files']:
        start = time.perf_counter()
        raw = (SOURCE / relative).read_bytes()
        if sha(raw) != plan['source_manifest'][relative]['sha256']:
            raise ValueError('Source changed')
        plain = gzip.decompress(raw)
        value = json.loads(plain)
        totals['source_read_seconds'] += time.perf_counter() - start
        start = time.perf_counter()
        transformed = value if mode.startswith('gzip') else encode(value, mode)
        encoded = canonical(transformed)
        compressed = gzip.compress(encoded, compresslevel=9 if mode == 'gzip9' else 3, mtime=0)
        destination = output / mode / f'repeat_{repeat}' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(compressed)
        totals['encode_seconds'] += time.perf_counter() - start
        start = time.perf_counter()
        readback = json.loads(gzip.decompress(destination.read_bytes()))
        restored = readback if mode.startswith('gzip') else decode(readback)
        restored_bytes = canonical(restored)
        if restored_bytes != plain:
            raise ValueError('Original canonical JSON bytes differ')
        totals['restore_seconds'] += time.perf_counter() - start
        candidates = value if isinstance(value, list) else value['candidates']
        records = sum(len(c[k]) for c in candidates for k in RECORD_KEYS if k in c)
        totals['episode_records'] += records
        totals['files'] += 1
        totals['original_stored_bytes'] += len(raw)
        totals['original_json_bytes'] += len(plain)
        totals['stored_bytes'] += len(compressed)
        totals['decoded_bytes'] += len(encoded)
        details.append({'path': relative, 'group': 'cost_fixture' if relative.startswith('cost_fixture_') else 'training_round',
                        'records': records, 'original_stored_bytes': len(raw), 'original_json_bytes': len(plain),
                        'stored_bytes': len(compressed), 'decoded_bytes': len(encoded), 'sha256': sha(compressed)})
    report = dict(mode=mode, repeat=repeat, totals=totals, files=details, all_original_bytes_equal=True,
                  peak_process_working_set_bytes=peak_memory_bytes(), policy_episode_calls=0)
    save_new(output / f'{mode}_{repeat}.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}), flush=True)


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    tests = selftest()
    files = sorted(p for p in SOURCE.rglob('*') if p.is_file())
    manifest = {p.relative_to(SOURCE).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())} for p in files}
    trajectory = [name for name in manifest if name.endswith('.json.gz')]
    if len(trajectory) != 122:
        raise ValueError('Expected all 120 rounds and two cost fixtures')
    schedule = []
    rng = random.Random(7100701)
    for repeat in range(3):
        order = list(MODES)
        rng.shuffle(order)
        schedule.extend({'mode': mode, 'repeat': repeat} for mode in order)
    plan = dict(experiment_id='Q1-010', created_utc=datetime.now(timezone.utc).isoformat(),
                purpose='기존 기록의 손실 없는 저장 비교', source='studies/interaction/results_q1007_v01/pilot',
                source_manifest=manifest, trajectory_files=trajectory, schedule=schedule,
                tool_sha256=sha(Path(__file__).read_bytes()),
                dependencies={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p.read_bytes()) for p in (
                    STUDY / 'evosim_interaction_v02/environment.py', STUDY / 'evosim_q1007_v01/resources.py')},
                python=sys.version, platform=platform.platform(), tests=tests,
                policy_episode_calls=0, timing='fresh child per arm/repeat; OS cache uncontrolled; encode includes disk write; restore includes read/hash/byte comparison',
                memory='child process lifetime peak, includes original and restored records for comparison',
                acceptance='all original decompressed canonical JSON bytes equal; all source bytes unchanged')
    save_new(output / 'plan.json', plan)
    for task in schedule:
        subprocess.run([sys.executable, '-B', '-u', '-X', 'utf8', str(Path(__file__).resolve()),
                        '--output', str(output), '--worker', task['mode'], '--repeat', str(task['repeat'])], check=True)
    reports = [json.loads((output / f"{t['mode']}_{t['repeat']}.json").read_bytes()) for t in schedule]
    unchanged = all(sha((SOURCE / name).read_bytes()) == item['sha256'] for name, item in manifest.items())
    if not unchanged:
        raise ValueError('Source files changed')
    summary = {'experiment_id': 'Q1-010', 'source_files': len(files), 'trajectory_files': len(trajectory),
               'policy_episode_calls': 0, 'all_source_files_unchanged': unchanged, 'tests': tests,
               'unchanged_sidecar_bytes': sum(v['bytes'] for n, v in manifest.items() if n not in trajectory),
               'plan_sha256': sha((output / 'plan.json').read_bytes()), 'arms': {}}
    for mode in MODES:
        group = [r for r in reports if r['mode'] == mode]
        totals = group[0]['totals']
        if len({r['totals']['stored_bytes'] for r in group}) != 1:
            raise ValueError('Nondeterministic size')
        summary['arms'][mode] = {**totals,
            'physical_reduction_fraction': 1 - totals['stored_bytes'] / totals['original_stored_bytes'],
            'logical_reduction_fraction': 1 - totals['decoded_bytes'] / totals['original_json_bytes'],
            'encode_seconds_runs': [r['totals']['encode_seconds'] for r in group],
            'restore_seconds_runs': [r['totals']['restore_seconds'] for r in group],
            'encode_seconds_median': statistics.median(r['totals']['encode_seconds'] for r in group),
            'restore_seconds_median': statistics.median(r['totals']['restore_seconds'] for r in group),
            'peak_memory_bytes_max': max(r['peak_process_working_set_bytes'] for r in group),
            'groups': {label: {key: sum(f[key] for f in group[0]['files'] if f['group'] == label)
                               for key in ('records', 'original_stored_bytes', 'original_json_bytes', 'stored_bytes', 'decoded_bytes')}
                       for label in ('training_round', 'cost_fixture')}}
    save_new(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--worker', choices=MODES)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        print(json.dumps(selftest()))
    elif args.worker:
        worker(args.output.resolve(), args.worker, args.repeat)
    else:
        main(args.output.resolve())
