"""Fixed-protocol, resumable experiments. Each worker owns its CONFIG and RNG.

Run: python run_research.py --workers 3
Raw seed-level outputs, policies and episode traces are written atomically.
The protocol and source digests are saved before any confirmatory test is read.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

from evosim import CONFIG, Environment, EvolutionarySimulator
from evosim.simulator import candidate_selection_key

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'research_results'
SEEDS = list(range(101, 111))
CONDITIONS = {
    'baseline': dict(q=.3, hebb=.015, edges=True, memory=True, init='seeded_positive'),
    'no_q_action': dict(q=0., hebb=.015, edges=True, memory=True, init='seeded_positive'),
    'no_plasticity': dict(q=.3, hebb=0., edges=True, memory=True, init='seeded_positive'),
    'neither_learning_path': dict(q=0., hebb=0., edges=True, memory=True, init='seeded_positive'),
    'edges_only': dict(q=.3, hebb=.015, edges=True, memory=False, init='seeded_positive'),
    'modules_only': dict(q=.3, hebb=.015, edges=False, memory=True, init='seeded_positive'),
    'no_memory': dict(q=.3, hebb=.015, edges=False, memory=False, init='seeded_positive'),
    'init_zero_mean': dict(q=.3, hebb=.015, edges=True, memory=True, init='zero_mean'),
    'init_no_edges': dict(q=.3, hebb=.015, edges=True, memory=True, init='none'),
    'edges_only_init_none': dict(q=.3, hebb=.015, edges=True, memory=False, init='none'),
}


@contextmanager
def config_override(**settings):
    old = dict(CONFIG)
    CONFIG.update(settings)
    try:
        yield
    finally:
        CONFIG.clear()
        CONFIG.update(old)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def policy(entity):
    return dict(
        modules={k: m.to_dict() for k, m in entity.modules.items()},
        connections={k: dict(v) for k, v in entity.connections.items()},
        recurrent_connections={k: dict(v) for k, v in entity.recurrent_connections.items()},
        q=[[list(state), action, value] for (state, action), value in entity.value_estimates.items()],
        allow_recurrent=entity.allow_recurrent,
        allow_recurrent_edges=entity.allow_recurrent_edges,
        allow_module_memory=entity.allow_module_memory,
    )


def frozen(entity, env, trajectories, seed, q=None, edges=None, memory=None, trace=True):
    """Freeze weights/Q; only transient activations may change. Record clustered data."""
    snap = entity.snapshot_for_eval()
    if edges is not None:
        snap.allow_recurrent_edges = edges
    if memory is not None:
        snap.allow_module_memory = memory
    with config_override(TD_WEIGHT=CONFIG['TD_WEIGHT'] if q is None else q):
        metrics = snap.evaluate_frozen(env, trajectories=trajectories, rng=random.Random(seed))
        if not trace:
            return metrics
        r = random.Random(seed)
        records = []
        for idx, trajectory in enumerate(trajectories):
            snap.reset_memory()
            steps = []
            for item in trajectory:
                key = snap._get_state_key(item['obs'])
                logits, _ = snap.forward(item['obs'])
                action, probs = snap.select_action(key, logits, rng=r)
                steps.append(dict(t=item['step'], masked=item['is_masked'],
                                  correct=action == item['correct_action'], action=action,
                                  reward=env.get_reward(action, probs, item['correct_action'])))
            records.append(dict(episode=idx, target=trajectory[0]['correct_action'], steps=steps))
        masked = [s['correct'] for e in records for s in e['steps'] if s['masked']]
        assert math.isclose(statistics.mean(masked), metrics['memory_accuracy'], abs_tol=1e-12)
        cue_ok = [e for e in records if e['steps'][0]['correct']]
        conditional = [s['correct'] for e in cue_ok for s in e['steps'] if s['masked']]
        metrics['memory_given_correct_cue'] = statistics.mean(conditional) if conditional else None
        metrics['episodes'] = records
        return metrics


def select(population, env, validation):
    scores = [e.evaluate_frozen(env, trajectories=validation, rng=random.Random(10001)) for e in population]
    idx = max(range(len(scores)), key=lambda i: (*candidate_selection_key(scores[i]), i))
    return idx


def run_one(name, seed):
    start = time.monotonic()
    condition = CONDITIONS[name]
    settings = dict(TD_WEIGHT=condition['q'], HEBBIAN_LR=condition['hebb'],
                    INITIAL_RECURRENCE_MODE=condition['init'], Q_STATE_INCLUDES_MEMORY=False)
    with config_override(**settings):
        random.seed(seed)
        sim = EvolutionarySimulator(
            allow_recurrent=True, allow_recurrent_edges=condition['edges'],
            allow_module_memory=condition['memory'], enable_validation=True,
            validation_scope='population', log_prefix=f'{name}_{seed}',
        )
        summary = sim.run(max_generations=100)
        env = sim.val_env
        test = env.generate_trajectories(120, seed=30003, shuffle_classes=True)
        result = dict(condition=name, seed=seed, configuration=condition,
                      summary=summary, generations=sim.logger.generation_data,
                      selected_generation=sim.best_val_generation + 1,
                      selected_stage=sim.best_val_trained_stage)
        result['historical'] = frozen(sim.best_val_entity, env, test, 30004)
        result['final'] = frozen(sim.final_gen_val_entity, env, test, 30004)
        # These are prospective secondary generalization sets, never used for selection.
        result['secondary_test_sets'] = {
            str(s): frozen(sim.best_val_entity, env, env.generate_trajectories(120, seed=s),
                           s + 100, trace=False) for s in range(30004, 30009)
        }
        i = select(sim.gen0_pre_learning_snapshots, env, sim.val_trajectories)
        j = select(sim.gen0_post_learning_snapshots, env, sim.val_trajectories)
        result['gen0_selected_pre'] = frozen(sim.gen0_pre_learning_snapshots[i], env, test, 30004)
        result['gen0_same_post'] = frozen(sim.gen0_post_learning_snapshots[i], env, test, 30004)
        result['gen0_selected_post'] = frozen(sim.gen0_post_learning_snapshots[j], env, test, 30004)
        result['gen0_lifetime_mean_delta'] = statistics.mean(
            frozen(post, env, test, 30004, trace=False)['memory_accuracy'] -
            frozen(pre, env, test, 30004, trace=False)['memory_accuracy']
            for pre, post in zip(sim.gen0_pre_learning_snapshots, sim.gen0_post_learning_snapshots)
        )
        if name == 'baseline':
            winner = sim.best_val_entity
            result['dependency'] = {
                'q_off': frozen(winner, env, test, 30004, q=0),
                'edges_off': frozen(winner, env, test, 30004, edges=False),
                'modules_off': frozen(winner, env, test, 30004, memory=False),
                'all_memory_off': frozen(winner, env, test, 30004, edges=False, memory=False),
                'pre_lifetime': frozen(sim.best_val_pre_lifetime_entity, env, test, 30004),
            }
            result['adaptation'] = {}
            adaptation_data = env.generate_trajectories(18, seed=40001)
            # Both branches start with the identical inherited policy and use identical
            # exogenous trajectories and RNG seed; original lifetime data are not replayed.
            for hebb in (0., .015):
                adapted = sim.best_val_pre_lifetime_entity.snapshot_for_eval()
                with config_override(HEBBIAN_LR=hebb):
                    state = random.getstate()
                    adapted.evaluate(env, learn=True, trajectories=adaptation_data, rng=random.Random(40002))
                    assert state == random.getstate()
                    for q in (0., .3):
                        result['adaptation'][f'hebb={hebb},q={q}'] = frozen(adapted, env, test, 30004, q=q)
        result['elapsed_seconds'] = time.monotonic() - start
        folder = OUT / name
        atomic_json(folder / f'{seed}_policies.json', dict(
            historical=policy(sim.best_val_entity),
            historical_pre_lifetime=policy(sim.best_val_pre_lifetime_entity),
            final=policy(sim.final_gen_val_entity),
            gen0_pre=policy(sim.gen0_pre_learning_snapshots[i]),
            gen0_same_post=policy(sim.gen0_post_learning_snapshots[i]),
        ))
        atomic_json(folder / f'{seed}.json', result)
        return name, seed, result['historical']['memory_accuracy'], result['elapsed_seconds']


def protocol():
    sources = list((ROOT / 'evosim').rglob('*.py')) + [Path(__file__)]
    return dict(version='research-1', seeds=SEEDS, conditions=CONDITIONS,
                generations=100, population=20, lifetime_episodes=18,
                validation_seed=10001, validation_episodes=30,
                primary_test_seed=30003, primary_test_episodes=120,
                secondary_test_seeds=list(range(30004, 30009)),
                adaptation_trajectory_seed=40001, adaptation_rng_seed=40002,
                candidate_rule='historical maximum of (validation fitness, accuracy, memory), latest tie',
                primary_endpoint='historical candidate frozen memory accuracy',
                secondary_endpoints=['final candidate', 'episode completion', 'time-specific accuracy',
                                     'conditional memory', 'secondary test seeds', 'paired adaptation'],
                caveats=['10 evolution seeds, one shared primary test set; no per-step independence',
                         'stage advancement and inherited learning are part of treatment',
                         'matched seed labels do not imply identical RNG paths across structures',
                         'mutation still seeds positive self-recurrence when adding modules',
                         'validation reused across 2000 candidates may overfit; test never selects',
                         'module-memory OFF maps ema/latch to tanh; it changes their instantaneous transfer too'],
                base_config=CONFIG, python=platform.python_version(),
                source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sources})


def describe(values):
    return dict(n=len(values), mean=statistics.mean(values), sd=statistics.stdev(values))


def bootstrap_ci(values, seed=51101, n=10000):
    r = random.Random(seed)
    means = sorted(statistics.mean(r.choices(values, k=len(values))) for _ in range(n))
    return [means[int(n*.025)], means[int(n*.975)]]


def summarize():
    data = {name: [json.loads((OUT / name / f'{s}.json').read_text(encoding='utf-8')) for s in SEEDS]
            for name in CONDITIONS}
    table = {}
    for name, runs in data.items():
        table[name] = {key: describe([r[phase][key] for r in runs])
                       for phase, key in [('historical', 'memory_accuracy'),
                                          ('historical', 'episode_full_memory_accuracy')]}
        table[name]['final_memory_accuracy'] = describe([r['final']['memory_accuracy'] for r in runs])
        table[name]['secondary_memory_accuracy'] = describe([
            statistics.mean(t['memory_accuracy'] for t in r['secondary_test_sets'].values()) for r in runs])
        table[name]['stage3_count'] = sum(r['summary']['final_trained_stage'] == 3 for r in runs)
    contrasts = {}
    for name in CONDITIONS:
        if name == 'baseline':
            continue
        differences = [r['historical']['memory_accuracy'] - b['historical']['memory_accuracy']
                       for r, b in zip(data[name], data['baseline'])]
        contrasts[f'{name}-baseline'] = dict(**describe(differences),
                                            paired_bootstrap_95=bootstrap_ci(differences))
    interaction = [data['baseline'][i]['historical']['memory_accuracy']
                   - data['no_q_action'][i]['historical']['memory_accuracy']
                   - data['no_plasticity'][i]['historical']['memory_accuracy']
                   + data['neither_learning_path'][i]['historical']['memory_accuracy']
                   for i in range(len(SEEDS))]
    contrasts['q_by_plasticity_interaction'] = dict(**describe(interaction), paired_bootstrap_95=bootstrap_ci(interaction))
    dependencies = {k: describe([r['dependency'][k]['memory_accuracy'] - r['historical']['memory_accuracy']
                                for r in data['baseline']]) for k in data['baseline'][0]['dependency']}
    adaptation = {k: describe([r['adaptation'][k]['memory_accuracy'] for r in data['baseline']])
                  for k in data['baseline'][0]['adaptation']}
    summary = dict(conditions=table, contrasts=contrasts, frozen_dependency_deltas=dependencies,
                   paired_adaptation=adaptation,
                   inference_note='Descriptive seed-level paired bootstrap intervals; not multiplicity-adjusted or causal isolation of inherited learning.')
    atomic_json(OUT / 'summary.json', summary)
    lines = ['condition,n,historical_mean,historical_sd,final_mean,episode_completion,secondary_mean,stage3_count']
    for name, row in table.items():
        lines.append(','.join(map(str, [name, 10, row['memory_accuracy']['mean'], row['memory_accuracy']['sd'],
                                       row['final_memory_accuracy']['mean'], row['episode_full_memory_accuracy']['mean'],
                                       row['secondary_memory_accuracy']['mean'], row['stage3_count']])))
    (OUT / 'summary.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    manifest = protocol()
    if (OUT / 'protocol.json').exists():
        old = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
        if old != manifest:
            raise RuntimeError('Protocol/source changed: preserve old outputs and use a new results directory.')
    else:
        atomic_json(OUT / 'protocol.json', manifest)
        env = Environment(CONFIG['CURRICULUM_STAGES'][-1])
        atomic_json(OUT / 'datasets.json', {str(s): env.generate_trajectories(120, seed=s)
                                          for s in range(30003, 30009)})
    if not args.summarize:
        jobs = [(name, s) for name in CONDITIONS for s in SEEDS if not (OUT / name / f'{s}.json').exists()]
        total = len(CONDITIONS) * len(SEEDS)
        complete = total - len(jobs)
        print(f'protocol saved; {complete}/{total} already complete', flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, name, seed): (name, seed) for name, seed in jobs}
            for future in as_completed(futures):
                name, seed, score, seconds = future.result()
                complete += 1
                print(f'{complete}/{total}: {name} seed={seed} memory={score:.4f} seconds={seconds:.1f}', flush=True)
    summarize()


if __name__ == '__main__':
    main()
