"""Distinguish removal of temporal state from replacing ema/latch by tanh."""
import hashlib
import json
from pathlib import Path
import random
import statistics

from evosim import CONFIG, Environment
from run_research import OUT, ROOT, SEEDS, atomic_json, bootstrap_ci, describe
from run_adaptation_followup import restore

DEST = ROOT / 'temporal_state_probe'
MODES = ['baseline', 'module_state_zero', 'legacy_module_off', 'edges_off', 'all_temporal_state_zero']


def measure(policy, env, data, mode):
    entity = restore(policy)
    if mode in ['edges_off', 'all_temporal_state_zero']:
        entity.allow_recurrent_edges = False
    if mode == 'legacy_module_off':
        entity.allow_module_memory = False
    if mode in ['module_state_zero', 'all_temporal_state_zero']:
        for module in entity.modules.values():
            original = module.compute
            def compute(weighted_sum, prev_self_state=0.0, allow_memory=True, original=original):
                return original(weighted_sum, prev_self_state=0.0, allow_memory=allow_memory)
            module.compute = compute
    selected = []
    original_select = entity.select_action
    def capture(key, logits, epsilon=0.0, rng=None):
        action, probs = original_select(key, logits, epsilon=epsilon, rng=rng)
        selected.append(action)
        return action, probs
    entity.select_action = capture
    _, _, _, _, metrics = entity.evaluate(env, learn=False, trajectories=data,
                                           rng=random.Random(50004), update_metrics=False, return_detailed=True)
    assert len(selected) == 4800
    metrics['episode_actions'] = [selected[i:i+4] for i in range(0, len(selected), 4)]
    return metrics


def main():
    DEST.mkdir(exist_ok=True)
    protocol = dict(classification='exploratory diagnostic of a confounded memory switch',
                    test_seed=50003, test_episodes=1200, seeds=SEEDS, modes=MODES,
                    module_state_zero='keep the original ema/latch equation and force its prev_self_state argument to zero',
                    selection='stored baseline historical candidates, no new selection',
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    prior_protocol_sha256=hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest())
    if (DEST / 'protocol.json').exists():
        assert json.loads((DEST / 'protocol.json').read_text(encoding='utf-8')) == protocol
    else:
        atomic_json(DEST / 'protocol.json', protocol)
    env = Environment(CONFIG['CURRICULUM_STAGES'][-1])
    data = env.generate_trajectories(1200, seed=50003)
    atomic_json(DEST / 'datasets.json', data)
    rows = []
    for seed in SEEDS:
        policy = json.loads((OUT / 'baseline' / f'{seed}_policies.json').read_text(encoding='utf-8'))['historical']
        row = dict(seed=seed, measurements={mode: measure(policy, env, data, mode) for mode in MODES})
        atomic_json(DEST / f'{seed}.json', row)
        rows.append(row)
    summary = {}
    for mode in MODES:
        values = [r['measurements'][mode]['memory_accuracy'] for r in rows]
        delta = [v-r['measurements']['baseline']['memory_accuracy'] for v,r in zip(values, rows)]
        summary[mode] = dict(**describe(values), memory_change=statistics.mean(delta),
                             paired_delta_bootstrap_95=bootstrap_ci(delta),
                             cue_mean=statistics.mean(r['measurements'][mode]['cue_accuracy'] for r in rows))
    atomic_json(DEST / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
