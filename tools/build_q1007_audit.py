"""기존 감사기를 새 실행 형태에 맞춰 처음 생성한다. 완료 원본은 보존한다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    old = ROOT / 'studies/interaction/evosim_q1006_v02/audit.py'
    new = ROOT / 'studies/interaction/evosim_q1007_v01/audit.py'
    if new.exists():
        raise FileExistsError(new)
    text = old.read_text(encoding='utf-8').split('\ndef audit_test_job(')[0]
    text = text.replace('from .search import directory, run_search, verify_candidate_lock', 'from .search import directory, run_search')
    text = text.replace('from .protocol import load', 'from .protocol import load, validation_episodes')
    text = text.replace('config = load(root)', 'config = load(root, condition)')
    text = text.replace('treatment = Treatment(condition)', 'treatment = Treatment("F0")')
    text = text.replace('validate_condition(genome, condition)', 'validate_condition(genome, "F0")')
    text = text.replace('rng, allocator, condition)', 'rng, allocator, "F0")')
    text = text.replace('derive_seed("initial", seed)', 'derive_seed("q1007:v01:initial", seed)')
    text = text.replace('derive_seed("search", seed, method)', 'derive_seed("q1007:v01:search", seed, method)')
    text = text.replace('derive_seed("training_batch", seed, generation)', 'derive_seed("q1007:v01:training_batch", seed, generation)')
    text = text.replace('for e in split.validation]', 'for e in validation_episodes(config)]')
    text = text.replace('    best = initial = final = None', '    best = initial = final = None\n    endpoints = {}')
    text = text.replace('        history = {', '        if str(generation + 1) in config["endpoint_rounds"]:\n            endpoints[config["endpoint_rounds"][str(generation + 1)]] = {"candidate": best, "through_round": generation + 1, "logical_episode_slots": calls}\n        history = {')
    text = text.replace('    if calls != result["logical_episode_slots"]', '    if endpoints != result["endpoints"] or endpoints != state["endpoints"]:\n        raise ValueError("끝점 후보가 해당 시점 누적 최선과 다름")\n    if calls != result["logical_episode_slots"]')
    new.write_text(text, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
