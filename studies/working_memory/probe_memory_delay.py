"""Exploratory out-of-training-horizon test; never re-selects candidates."""
import hashlib
import json
from pathlib import Path
import statistics

from evosim import CONFIG, Environment
from run_research import CONDITIONS, OUT, ROOT, SEEDS, atomic_json, config_override, describe, frozen
from run_adaptation_followup import restore

DEST = ROOT / 'delay_probe'
NAMES = ['baseline', 'neither_learning_path', 'edges_only', 'modules_only', 'no_memory', 'edges_only_init_none']
DELAYS = [1, 3, 5, 8, 16, 32]


def main():
    DEST.mkdir(exist_ok=True)
    protocol = dict(classification='exploratory after primary study; no new candidate selection',
                    conditions=NAMES, delays=DELAYS, seeds=SEEDS, episodes=120,
                    environment_seed=50001, action_seed=50002,
                    design='Generate one 33-step trajectory per episode; all delays use its prefix, preserving the exact cue and earlier mask noise.',
                    endpoint='accuracy of the response at the last masked step, seed-level mean and sd',
                    prior_protocol_sha256=hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (DEST / 'protocol.json').exists():
        if json.loads((DEST / 'protocol.json').read_text(encoding='utf-8')) != protocol:
            raise RuntimeError('Delay protocol changed.')
    else:
        atomic_json(DEST / 'protocol.json', protocol)
    with config_override(STEPS_PER_EPISODE=33):
        env = Environment(dict(noise=.15, mask_steps=32, threshold=.7))
        longest = env.generate_trajectories(120, seed=50001)
    atomic_json(DEST / 'datasets.json', longest)
    rows = []
    for name in NAMES:
        for seed in SEEDS:
            saved = json.loads((OUT / name / f'{seed}_policies.json').read_text(encoding='utf-8'))
            entity = restore(saved['historical'])
            measurements = {}
            for delay in DELAYS:
                prefix = [episode[:delay+1] for episode in longest]
                result = frozen(entity, env, prefix, 50002, q=CONDITIONS[name]['q'])
                episodes = result.pop('episodes')
                result['episode_actions'] = [[s['action'] for s in e['steps']] for e in episodes]
                result['last_masked_accuracy'] = result['step_accuracies'][delay]
                measurements[str(delay)] = result
            row = dict(condition=name, seed=seed, measurements=measurements)
            rows.append(row)
            atomic_json(DEST / f'{name}_{seed}.json', row)
        print(f'delay probe {name} complete', flush=True)
    summary = {name: {str(delay): describe([
        r['measurements'][str(delay)]['last_masked_accuracy'] for r in rows if r['condition'] == name
    ]) for delay in DELAYS} for name in NAMES}
    atomic_json(DEST / 'summary.json', summary)
    lines = ['condition,' + ','.join(f'last_step_d{d}' for d in DELAYS)]
    for name in NAMES:
        lines.append(name + ',' + ','.join(str(summary[name][str(d)]['mean']) for d in DELAYS))
    (DEST / 'summary.csv').write_text('\n'.join(lines)+'\n', encoding='utf-8-sig')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
