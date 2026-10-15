"""Prospectively fixed exploratory test of the TD bootstrap in plasticity.

Uses the same inherited pre-lifetime candidate in all branches, a common
training prefix, and a fresh 1200-episode test (30009). No re-selection.
"""
import hashlib
import json
from pathlib import Path
import random
import statistics

from evosim import CONFIG, Entity, Environment
from run_research import OUT, ROOT, SEEDS, atomic_json, bootstrap_ci, config_override, describe, frozen

DEST = ROOT / 'adaptation_followup'


def restore(payload):
    genome = {k: payload[k] for k in ['modules', 'connections', 'recurrent_connections']}
    values = {(tuple(state), action): value for state, action, value in payload['q']}
    return Entity(genome, value_estimates=values, decay_values=False,
                  allow_recurrent=payload['allow_recurrent'],
                  allow_recurrent_edges=payload['allow_recurrent_edges'],
                  allow_module_memory=payload['allow_module_memory'])


def measure(entity, env, test, q):
    result = frozen(entity, env, test, 41003, q=q)
    episodes = result.pop('episodes')
    result['episode_actions'] = [[s['action'] for s in e['steps']] for e in episodes]
    return result


def main():
    DEST.mkdir(exist_ok=True)
    sources = list((ROOT / 'evosim').rglob('*.py')) + [Path(__file__), ROOT / 'run_research.py']
    manifest = dict(
        classification='exploratory follow-up chosen after primary results, fixed before new test',
        seeds=SEEDS, discount=[0.0, 0.9], hebbian_lr=[0.0, 0.015], training_episodes=[18, 180],
        adaptation_action_q_weight=0.3, test_q_weights=[0.0, 0.3],
        train_seed=41001, adaptation_rng_seed=41002, test_seed=30009, test_episodes=1200,
        primary_endpoint='post-minus-pre frozen memory accuracy with Q-action OFF',
        selection='baseline historical winner inherited pre-lifetime policy, no new candidate selection',
        training_prefix='same prefix of a shuffled balanced 180-episode Stage 3 dataset',
        caveats=['candidate conditioned on original post-learning validation selection',
                 'no inherited-weight reset; this is incremental plasticity, not origin of all learning',
                 'Q-action ON during adaptation in all branches',
                 '18-prefix class counts need not be exactly balanced'],
        original_protocol_sha256=hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    )
    if (DEST / 'protocol.json').exists():
        if json.loads((DEST / 'protocol.json').read_text(encoding='utf-8')) != manifest:
            raise RuntimeError('Follow-up protocol changed.')
    else:
        atomic_json(DEST / 'protocol.json', manifest)
    env = Environment(CONFIG['CURRICULUM_STAGES'][-1])
    training = env.generate_trajectories(180, seed=41001)
    test = env.generate_trajectories(1200, seed=30009)
    atomic_json(DEST / 'datasets.json', dict(training=training, test=test))
    rows = []
    for seed in SEEDS:
        path = DEST / f'{seed}.json'
        if path.exists():
            rows.append(json.loads(path.read_text(encoding='utf-8')))
            continue
        saved = json.loads((OUT / 'baseline' / f'{seed}_policies.json').read_text(encoding='utf-8'))
        pre = restore(saved['historical_pre_lifetime'])
        row = dict(seed=seed, before={str(q): measure(pre, env, test, q) for q in (0.0, 0.3)}, branches={})
        for episodes in (18, 180):
            for gamma in (0.0, 0.9):
                for hebb in (0.0, 0.015):
                    adapted = pre.snapshot_for_eval()
                    with config_override(TD_WEIGHT=.3, TD_DISCOUNT=gamma, HEBBIAN_LR=hebb):
                        state = random.getstate()
                        adapted.evaluate(env, learn=True, trajectories=training[:episodes], rng=random.Random(41002))
                        assert state == random.getstate()
                        scores = {str(q): measure(adapted, env, test, q) for q in (0.0, 0.3)}
                        if hebb == 0.0:
                            assert scores['0.0']['episode_actions'] == row['before']['0.0']['episode_actions']
                    row['branches'][f'n={episodes},gamma={gamma},hebb={hebb}'] = scores
        atomic_json(path, row)
        rows.append(row)
        print(f'adaptation follow-up seed {seed} complete', flush=True)
    summary = {}
    for branch in rows[0]['branches']:
        delta = [r['branches'][branch]['0.0']['memory_accuracy'] - r['before']['0.0']['memory_accuracy'] for r in rows]
        summary[branch] = dict(**describe(delta), paired_bootstrap_95=bootstrap_ci(delta),
                               memory_mean=statistics.mean(r['branches'][branch]['0.0']['memory_accuracy'] for r in rows))
    contrasts = {}
    for episodes in (18, 180):
        a, b = f'n={episodes},gamma=0.0,hebb=0.015', f'n={episodes},gamma=0.9,hebb=0.015'
        differences = [r['branches'][a]['0.0']['memory_accuracy'] - r['branches'][b]['0.0']['memory_accuracy'] for r in rows]
        contrasts[str(episodes)] = dict(**describe(differences), paired_bootstrap_95=bootstrap_ci(differences))
    atomic_json(DEST / 'summary.json', dict(branches=summary, immediate_minus_bootstrap=contrasts,
                                          inference_note='Exploratory, unadjusted descriptive paired bootstrap intervals, 10 selected candidates.'))
    print(json.dumps(dict(branches=summary, immediate_minus_bootstrap=contrasts), indent=2), flush=True)


if __name__ == '__main__':
    main()
