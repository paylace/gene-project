"""완료 소스를 수정하지 않고 Q1-007의 계승 파일과 탐색기를 처음 생성한다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'studies/interaction/evosim_q1006_v02'
NEW = ROOT / 'studies/interaction/evosim_q1007_v01'


def main():
    NEW.mkdir(exist_ok=False)
    for name in ('__init__.py', 'network.py', 'model.py', 'evaluation.py'):
        (NEW / name).write_bytes((OLD / name).read_bytes())
    text = (OLD / 'search.py').read_text(encoding='utf-8').split('\ndef candidate_manifest(')[0]
    text = text.replace('Q1-006 훈련 점수', 'Q1-007 끝점별 후보와 훈련 점수')
    text = text.replace('config = load(root)', 'config = load(root, condition)')
    text = text.replace('"experiment_id": config["experiment_id"], "phase"', '"experiment_id": config["experiment_id"], "phase"')
    text = text.replace('best, initial, initial_population = state["best"], state["initial"], state["initial_population"]',
                        'best, initial, initial_population = state["best"], state["initial"], state["initial_population"]\n        endpoints = state["endpoints"]')
    text = text.replace('best = initial = None', 'best = initial = None\n        endpoints = {}')
    text = text.replace('derive_seed("initial", seed)', 'derive_seed("q1007:v01:initial", seed)')
    text = text.replace('derive_seed("search", seed, method)', 'derive_seed("q1007:v01:search", seed, method)')
    text = text.replace('derive_seed("training_batch", seed, generation)', 'derive_seed("q1007:v01:training_batch", seed, generation)')
    text = text.replace('validate_condition(genome, condition)', 'validate_condition(genome, "F0")')
    text = text.replace('rng, allocator, condition)', 'rng, allocator, "F0")')
    text = text.replace('f"train:{generation}", treatment', 'f"train:{generation}:q1007:v01:slot{slot}", treatment')
    text = text.replace('"validation", treatment, repeats=3', 'f"validation:q1007:v01:slot{slot}", treatment, repeats=config["repeats"]')
    text = text.replace('split.validation, seed, f"validation', 'validation_episodes(config), seed, f"validation')
    text = text.replace('check_access(config, "validation", split.validation)', 'check_access(config, "validation", validation_episodes(config))')
    text = text.replace('from .protocol import check_access, load', 'from .protocol import check_access, load, validation_episodes')
    text = text.replace('size * (24 + (252 * 3 if generation in config["validation_rounds"] else 0))',
                        'size * (24 + (len(validation_episodes(config)) * config["repeats"] if generation in config["validation_rounds"] else 0))')
    text = text.replace('maximum_episode_calls": size', 'maximum_single_step_actions": size * 24 * (24 + (len(validation_episodes(config)) * config["repeats"] if generation in config["validation_rounds"] else 0)),\n                   "maximum_episode_calls": size')
    text = text.replace('attempt.update(status="evaluated", completed_episode_calls=episode_calls - calls_before)',
                        'attempt.update(status="evaluated", completed_episode_calls=episode_calls - calls_before,\n                       completed_single_step_actions=sum(len(r["actions"]) for row in rows for field in ("training_records", "validation_records") for r in row.get(field, [])))')
    text = text.replace('        history.append(', '        if str(generation + 1) in config["endpoint_rounds"]:\n            label = config["endpoint_rounds"][str(generation + 1)]\n            endpoints[label] = {"candidate": best, "through_round": generation + 1, "logical_episode_slots": episode_calls}\n        history.append(')
    text = text.replace('"best": best, "initial": initial, "initial_population"', '"endpoints": endpoints, "best": best, "initial": initial, "initial_population"')
    text = text.replace('"identity": identity, "best": best, "initial": initial, "final": final_candidate,',
                        '"identity": identity, "endpoints": endpoints, "best": best, "initial": initial, "final": final_candidate,')
    (NEW / 'search.py').write_text(text, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
