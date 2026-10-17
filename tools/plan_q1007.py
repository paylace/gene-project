"""Q1-007 실행 전 예산·가정별 판정 확률 계산. 정책을 실행하지 않는다."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/protocols/q1_007_v01_draft.json"
OUTPUT = ROOT / "docs/management/q1_007_design_preview.json"


def binomial_p(n, k, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def exact_p(wins, losses, two_sided=False):
    discordant = wins + losses
    if not discordant:
        return 1.0
    boundary = max(wins, losses) if two_sided else wins
    tail = sum(math.comb(discordant, k) for k in range(boundary, discordant + 1)) / 2 ** discordant
    return min(1.0, tail * (2 if two_sided else 1))


def passing_probability(n, p_control, p_treatment, p_both, two_sided):
    # 두 조건 모두 성공하거나 모두 실패한 시드는 대응 비교에서 동률이다.
    # 두 조건의 성공확률과 공동 성공확률은 관측값이 아닌 설계용 가정이다.
    win, loss = p_treatment - p_both, p_control - p_both
    if min(win, loss, p_both, 1 - p_control - p_treatment + p_both) < 0:
        raise ValueError("유효하지 않은 공동 확률")
    q = win + loss
    if q == 0:
        return 0.0
    probability = 0.0
    for d in range(n + 1):
        pd = binomial_p(n, d, q)
        for w in range(d + 1):
            delta = (2 * w - d) / n
            practical = abs(delta) if two_sided else delta
            if practical + 1e-12 >= .10 and exact_p(w, d - w, two_sided) <= .025 + 1e-12:
                probability += pd * binomial_p(d, w, win / q)
    return probability


def endpoint_budget(population, rounds, spec):
    interval = spec["validation_every_rounds"]
    if rounds % interval or spec["validation_at_round0"]:
        raise ValueError("10회차 검증 간격과 맞지 않음")
    genomes = population * rounds
    return {"population": population, "rounds": rounds,
            "training_genome_slots": genomes,
            "training_episode_slots": genomes * spec["training_batch"],
            "validation_candidate_slots": genomes // interval,
            "validation_episode_slots": genomes // interval * spec["validation_episodes"] * spec["evaluation_repeats"],
            "search_episode_slots": genomes * spec["training_batch"] + genomes // interval * spec["validation_episodes"] * spec["evaluation_repeats"]}


def main():
    spec = json.loads(PLAN.read_text(encoding="utf-8"))
    endpoints = {name: endpoint_budget(*shape, spec) for name, shape in spec["evolution_endpoints"].items()}
    random = {name: endpoint_budget(20, count // 20, spec) for name, count in spec["random_endpoints"].items()}
    assert endpoints["P"]["search_episode_slots"] == endpoints["T"]["search_episode_slots"]
    assert endpoints["P"]["validation_candidate_slots"] == endpoints["T"]["validation_candidate_slots"]
    assert endpoints["PT"]["training_genome_slots"] == 4 * endpoints["B"]["training_genome_slots"]
    assert len(spec["main_seeds"]) == len(set(spec["main_seeds"])) == 80
    assert not set(spec["main_seeds"]) & set(spec["confirmation_seeds"])
    assert exact_p(6, 0) == .015625 and exact_p(6, 0, True) == .03125
    assert exact_p(0, 0) == 1 and exact_p(0, 6) == 1
    per_seed = endpoints["T"]["search_episode_slots"] + endpoints["PT"]["search_episode_slots"] + random["R8000"]["search_episode_slots"]
    n = len(spec["main_seeds"])
    logical = sum(row["search_episode_slots"] for row in list(endpoints.values()) + list(random.values())) * n
    scenarios = []
    for sample_size in (40, 80, 120):
        for pc, pt, pb in ((.10, .10, .025), (.10, .20, .025), (.10, .25, .025),
                           (.10, .25, .05), (.10, .25, .10), (.10, .35, .05)):
            scenarios.append({"paired_seeds": sample_size, "p_control": pc, "p_treatment": pt,
                              "p_both": pb,
                              "one_sided_gate_probability": passing_probability(sample_size, pc, pt, pb, False),
                              "two_sided_gate_probability": passing_probability(sample_size, pc, pt, pb, True)})
    result = {
        "experiment_id": "Q1-007", "status": "실행 전 설계 계산; 본·예비 미실행",
        "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        "policy_episode_calls": 0, "network_forward_calls": 0,
        "evolution_endpoints": endpoints, "random_endpoints": random,
        "main": {"paired_seeds": n, "physical_search_runs": n * 3,
                 "logical_endpoint_results": n * 7,
                 "search_episode_slots_with_prefix_reuse": per_seed * n,
                 "search_episode_slots_without_prefix_reuse": logical,
                 "saved_search_episode_slots": logical - per_seed * n,
                 "test_episode_slots_upper_bound": n * 7 * 504 * 3},
        "confirmation_search_slots_max": n * (endpoints["T"]["search_episode_slots"] + endpoints["PT"]["search_episode_slots"]),
        "confirmation_test_slots_max": n * 4 * 504 * 3,
        "pilot_training_episode_slots": len(spec["pilot_seeds"]) * sum(size for _, size in spec["pilot_runs_per_seed"]) * spec["pilot_rounds"] * spec["training_batch"],
        "additional_cost_fixture_episode_slots_max": spec["cost_gate"]["extra_cost_fixture_episode_slots_max"],
        "power_scope": "각 대비 p<=.025의 보수적 문턱과 관측 차이 10pp를 결합한 정확 열거 확률. 실제 Holm 전체 검출력 아님. 가정이며 실험 결과 아님.",
        "power_scenarios": scenarios, "independent_review": False}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "policy_episode_calls", "main", "pilot_training_episode_slots")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
