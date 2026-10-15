"""Q1-004 초안 검사와 작업/비용 미리보기. 탐색·정책 평가·잠금 기능 없음."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs/protocols/q1_004_v02_draft.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_digest(value):
    data = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("ascii")).hexdigest()


def local_path(value, root=ROOT):
    require(isinstance(value, str) and not Path(value).is_absolute() and "\\" not in value,
            "저장소 상대 POSIX 경로가 필요합니다")
    path = (root / value).resolve()
    require(path.is_relative_to(root.resolve()), "저장소 밖 경로는 허용하지 않습니다")
    return path


def validate(config, root=ROOT):
    require(config["schema_version"] == 1 and config["plan_id"] == "Q1-004", "지원하지 않는 계획")
    require(config["status"] == "draft" and config["locked"] is False
            and config["implementation_source_digest"] is None,
            "이 도구는 미잠금 계획 초안만 검사합니다. 실행 프로토콜 검증기가 아닙니다")
    for path in config["paths"].values():
        local_path(path, root)
    require(config["paths"] == {"implementation": "studies/interaction/evosim_interaction_v02",
                                "results": "studies/interaction/results_v02"}, "신규 버전 경로 불일치")
    baseline = config["baseline"]
    original = json.loads(local_path(baseline["protocol"], root).read_text(encoding="utf-8"))
    require(original["source_digest"] == baseline["source_digest"], "기준 소스 식별자 불일치")
    actual_manifest = {}
    for name, digest in original["sources"].items():
        source = local_path(baseline["source_root"] + "/" + name.replace("\\", "/"), root)
        actual_manifest[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        require(actual_manifest[name] == digest, "동결 소스 변경: " + name)
    old_digest = hashlib.sha256(json.dumps(actual_manifest, sort_keys=True).encode()).hexdigest()
    require(old_digest == baseline["source_digest"], "동결 매니페스트 해시 불일치")
    # 동결 정의에서 목록만 생성한다. 정책/평가/탐색 모듈은 가져오지 않는다.
    package_root = str(local_path("studies/interaction", root))
    sys.path.insert(0, package_root)
    try:
        datasets = importlib.import_module("evosim_interaction.datasets")
        policies = importlib.import_module("evosim_interaction.policies")
        split = datasets.make_split(config["dataset"]["seed"])
        manifest = split.manifest()
        patterns = policies.blind_patterns(config["auxiliary"]["blind_seed"])
    finally:
        sys.path.pop(0)
    require(split.digest == config["dataset"]["digest"] == original["split_digest"], "분할 해시 불일치")
    sizes = {key: len(values) for key, values in manifest.items()}
    require(sizes == config["dataset"]["sizes"], "분할 크기 불일치")
    parts = [set(values) for values in manifest.values()]
    require(sum(map(len, parts)) == sum(sizes.values())
            and all(not a & b for i, a in enumerate(parts) for b in parts[i + 1:]), "분할 중복")
    require(config["dataset"]["previously_observed_benchmark"] is True, "벤치마크 선행 노출 표시 필요")
    expected_conditions = {
        "A0": dict(output_self_recurrence=True, other_recurrence=True, module_memory=True),
        "A1": dict(output_self_recurrence=False, other_recurrence=True, module_memory=True),
        "A2": dict(output_self_recurrence=False, other_recurrence=False, module_memory=False)}
    require(config["conditions"] == expected_conditions, "A0/A1/A2 정의 불일치")
    operators = config["operators"]
    require(operators["proposal_weights"] == original["settings"]["mutation_weights"], "변이 가중치 변경")
    require(operators["kinds"] == ["weight", "add_module", "remove_module", "add_ff", "add_rec", "remove_edge"],
            "연산 순서 변경")
    for key, expected in {
        "forbidden_proposal": "no_op_no_resampling",
        "forbidden_module_mutation": "retain_current_op",
        "forbidden_new_module_op": "replace_with_tanh",
        "remove_edge_mapping": "retain_original_0.4_rec_draw",
        "forbidden_add_rec_draws": "draw_target_source_and_if_absent_weight_then_reject",
    }.items():
        require(operators[key] == expected, "지원하지 않는 제안 규칙: " + key)
    search = config["search"]
    require(search["methods"] == ["evolution", "random"] and search["roles"] == ["best", "initial", "final"],
            "비교 방법/후보 역할 불일치")
    require(search["population"] == 20 and search["training_batch"] == 24
            and search["training_repeats"] == 1 and search["evaluation_repeats"] == 3, "공통 평가 예산 변경")
    require(search["selection"] == ["success_rate", "mean_reward"]
            and search["ties"] == "first_generation_then_first_slot"
            and search["rng_recipe"] == "v01_derive_seed_no_condition_name", "선발/난수 규칙 불일치")
    require(search["validation_rounds"] == [0] + list(range(9, 100, 10)), "검증 회차 불일치")
    seen = set()
    for name, phase in config["phases"].items():
        seeds = phase["seeds"]
        require(all(type(seed) is int for seed in seeds) and len(set(seeds)) == len(seeds), "시드 중복/형식 오류")
        require(not seen.intersection(seeds), "단계 간 시드 중복")
        seen.update(seeds)
        pilot = name == "pilot"
        require(name in ("pilot", "main", "confirmation"), "알 수 없는 단계")
        require(len(seeds) == (2 if pilot else 10) and phase["rounds"] == (5 if pilot else 100), "회차/시드 수 변경")
        require(phase["validation"] is (not pilot) and phase["test"] is (not pilot), "예비의 검증/테스트 접근 금지")
        require(phase["conditions"] == (["A0", "SELECTED"] if name == "confirmation" else ["A0", "A1", "A2"]),
                "단계 조건 누락/중복")
    require(set(config["phases"]) == {"pilot", "main", "confirmation"}, "필수 단계 누락")
    analysis = config["analysis"]
    require(analysis["unit"] == "search_seed" and analysis["multiplicity"] == "Holm_two_primary_contrasts"
            and analysis["zero_difference"] == "nonpositive"
            and analysis["test"] == "one_sided_exact_binomial_sign", "주 분석 정의 불일치")
    require(analysis["primary"] == ["A1_evolution_minus_A0_evolution", "A2_evolution_minus_A0_evolution"]
            and analysis["confirmation_priority"] == ["A1", "A2"], "대비/후속 선택 불일치")
    for key, expected in {
        "bootstrap_seed": 72004, "bootstrap_samples": 10000, "percentiles": [2.5, 97.5],
        "alpha": 0.05, "minimum_mean_difference": 0.1, "minimum_positive_seeds": 8,
        "probe_threshold": 0.8, "minimum_probe_passing_seeds": 8,
        "secondary_status": "exploratory_unadjusted",
    }.items():
        require(analysis[key] == expected, "분석 기준 변경: " + key)
    require(config["auxiliary"] == {
        "blind_seed": 71002, "blind_validation_repeats": 1, "uniform_coordinate_test_seed": 71003,
        "uniform_test_repeats": 20, "probe_repeats": 3, "diagnostic_roles": ["best"],
        "confirmation_reuse_fixed_baselines": True, "action_budget": 24,
    }, "보조 평가 정의 불일치")
    require(set(config["lock_requirements"]) == {
        "implementation_and_analysis_source_manifest", "immutable_resolved_settings_per_condition",
        "a0_equivalence_and_invariant_test_results", "pilot_cost_and_resume_report",
        "datasets_blind_patterns_and_probe_manifest", "python_numpy_platform_and_exact_commands",
        "protocol_digest_and_checkpoint_identity_schema",
    } and len(config["lock_requirements"]) == 7, "잠금 준비 항목 누락/중복")
    return original, manifest, patterns


def preview(config, root=ROOT):
    original, manifest, patterns = validate(config, root)
    search, sizes = config["search"], config["dataset"]["sizes"]
    phases, jobs = {}, []
    for name, phase in config["phases"].items():
        count = len(phase["seeds"]) * len(phase["conditions"]) * len(search["methods"])
        training_slots = count * search["population"] * phase["rounds"]
        validation_slots = count * search["population"] * len(search["validation_rounds"]) if phase["validation"] else 0
        training = training_slots * search["training_batch"] * search["training_repeats"]
        validation = validation_slots * sizes["validation"] * search["evaluation_repeats"]
        test = count * len(search["roles"]) * sizes["test"] * search["evaluation_repeats"] if phase["test"] else 0
        require(training + validation == phase["expected_train_validation"], "예산 산식 불일치: " + name)
        phases[name] = dict(jobs=count, training_slots=training_slots, validation_slots=validation_slots,
                            training_episodes=training, validation_episodes=validation, test_role_episodes=test)
        for condition in phase["conditions"]:
            for method in search["methods"]:
                for seed in phase["seeds"]:
                    jobs.append(dict(phase=name, condition=condition, method=method, seed=seed,
                                     conditional=name == "confirmation",
                                     planned_directory=f"{config['paths']['results']}/{name}/{condition}/{method}_{seed}"))
    aux = config["auxiliary"]
    probe_pairs = [(x, left, right) for x in range(1, 6) for left in range(x) for right in range(x + 1, 7)]
    baseline_validation = len(patterns) * sizes["validation"] * aux["blind_validation_repeats"]
    baseline_test = sizes["test"] * (aux["uniform_test_repeats"] + 2)
    for name in ("main", "confirmation"):
        phases[name]["probe_pair_trials"] = phases[name]["jobs"] * len(probe_pairs) * aux["probe_repeats"]
        phases[name]["probe_forward_calls"] = phases[name]["probe_pair_trials"] * 2
    return {
        "plan_id": config["plan_id"], "plan_revision": config["plan_revision"],
        "kind": "planning_preview_not_execution_protocol", "locked": False, "experiments_executed": 0,
        "plan_digest": canonical_digest(config), "baseline_source_files_verified": len(original["sources"]),
        "resolved_baseline_settings": original["settings"],
        "mutation_weight_sum": sum(config["operators"]["proposal_weights"]),
        "mutation_probabilities": [w / sum(config["operators"]["proposal_weights"])
                                   for w in config["operators"]["proposal_weights"]],
        "datasets": manifest, "blind_patterns": patterns, "probe_pairs": probe_pairs,
        "phases": phases, "jobs": jobs,
        "auxiliary_budget": {"shared_blind_validation_episodes": baseline_validation,
                             "shared_baseline_test_episodes": baseline_test,
                             "main_counterfactual_extra_forward_calls_upper_bound":
                                 phases["main"]["jobs"] * sizes["test"] * search["evaluation_repeats"] * aux["action_budget"] * 2,
                             "verification_episodes": "implementation_pending_measured_separately"},
        "total_planned_episodes_including_conditional_confirmation_and_shared_baselines":
            sum(p["training_episodes"] + p["validation_episodes"] + p["test_role_episodes"] for p in phases.values())
            + baseline_validation + baseline_test,
        "execution_ready": False, "unmet_lock_requirements": config["lock_requirements"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = preview(json.loads(args.config.read_text(encoding="utf-8")))
    # 표준 출력만 사용한다. 프로토콜이나 결과 디렉터리를 생성하지 않는다.
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
