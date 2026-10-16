"""Q1-008 원본을 변경하지 않고 수치 재생·측정 대조를 별도 보관한다."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/interaction"
sys.path.insert(0, str(STUDY))
SCRIPT = STUDY / "diagnostics_q1008_v01/diagnose_initial_reactivity.py"
OUT = STUDY / "diagnostics_q1008_review_v01/results.json"


def main():
    if OUT.exists():
        raise FileExistsError("Review evidence already exists")
    spec = importlib.util.spec_from_file_location("original_q1008", SCRIPT)
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    saved = json.loads((SCRIPT.parent / "results.json").read_text(encoding="utf-8"))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    guards = {name: sha(STUDY / "evosim_interaction_v02" / name) == digest
              for name, digest in saved["frozen_source_sha256"].items()}
    guards["diagnostic_script"] = sha(SCRIPT) == saved["script_sha256"]
    assert all(guards.values())
    episodes = diagnostic.all_episodes()
    sample = random.Random(0).sample(episodes, 200)[:40]
    assert [e.episode_id for e in sample] == saved["parameters"]["episode_ids"]
    reproduced = [diagnostic.diagnose(c, sample) for c in ("A0", "A1", "A2")]
    reproduction = {row["condition"]: row == old for row, old in zip(reproduced, saved["conditions"])}
    print("원 진단 집계 재생:", reproduction, flush=True)

    class Coordinate:
        def reset(self):
            pass
        def act(self, observation, rng):
            p, g = observation[:2]
            return 2 if p == g else (0 if g < p else 1)

    class ZeroLogits:
        settings = diagnostic.Settings("A0")
        def reset(self):
            pass
        def forward(self, observation):
            return [0.0, 0.0, 0.0]
        act = diagnostic.NetworkPolicy.act

    coordinate = diagnostic.first_step(Coordinate())
    zero = diagnostic.first_step(ZeroLogits())
    pairs = [(p, g, h) for p in range(7) for g in range(7) if p != g
             for h in (g-1, g+1) if 0 <= h < 7 and h != p]
    same_label = sum(diagnostic.correct_action(p, g) == diagnostic.correct_action(p, h) for p,g,h in pairs)
    class NearLogits(ZeroLogits):
        def forward(self, observation):
            return [900.0000005, 900.0, 0.0]
    logits = NearLogits().forward(None)
    choices = [i for i,v in enumerate(logits) if abs(v-max(logits)) <= 1e-9*max(1,abs(max(logits)))]
    synthetic = {"logits": logits, "original_diagnostic_action": random.Random(0).choice(choices),
                 "actual_policy_action": NearLogits().act(None, random.Random(0))}

    class CapturingPolicy(diagnostic.NetworkPolicy):
        def forward(self, observation):
            result = super().forward(observation)
            self.last_logits = result
            return result

    def actual_rollout(policy, episode):
        env = diagnostic.LineCollectEnv()
        observation = env.reset(episode)
        policy.reset()
        rng = random.Random(0)
        history, peak, correct, steps = [], 0.0, 0, 0
        while True:
            state = env.state
            action = policy.act(observation, rng)
            peak = max(peak, max(abs(v) for v in policy.last_logits))
            correct += action == diagnostic.correct_action(state.position, state.goal)
            steps += 1
            result = env.step(action)
            history.append((action, result.failed_collect, result.wall_collision))
            if result.terminated:
                return history, peak, correct, steps, env.state.collected
            observation = result.observation

    diagnostic.NetworkPolicy = CapturingPolicy
    diagnostic.rollout = actual_rollout
    corrected = [diagnostic.diagnose(c, sample) for c in ("A0", "A1", "A2")]
    actual_agreement = {row["condition"]: row == old for row, old in zip(corrected, reproduced)}
    print("실제 act 경로 집계 일치:", actual_agreement, flush=True)
    # 수작업 해를 진화의 모듈 수·가중치 범위 안에서도 구성할 수 있는지 별도 확인한다.
    settings = diagnostic.Settings("A2")
    genome = diagnostic.Genome(
        modules={f"m{i}": diagnostic.Module("linear", 0.0, 0.5) for i in range(4)},
        ff={"out_0": {"in_0": 2.0, "in_1": -2.0},
            "out_1": {"in_0": -2.0, "in_1": 2.0}, "out_2": {"m0": 0.5}},
        rec={}, settings=settings)
    policy = CapturingPolicy(genome, settings)
    successes = optimal = 0
    for episode in episodes:
        history, _, _, steps, collected = actual_rollout(policy, episode)
        successes += collected == 3
        optimal += collected == 3 and steps == episode.optimal_steps
    output = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "review_script_sha256": sha(Path(__file__)), "source_guards": guards,
        "original_aggregate_reproduction": reproduction, "reproduced_conditions": reproduced,
        "actual_act_aggregate_agreement": actual_agreement, "actual_act_conditions": corrected,
        "adjacent_pairs": len(pairs), "adjacent_pairs_with_same_correct_action": same_label,
        "coordinate_first_step": dict(zip(("correct", "states", "responsive", "flips", "adjacent"), coordinate)),
        "zero_logits_first_step": dict(zip(("correct", "states", "responsive", "flips", "adjacent"), zero)),
        "tie_rule_counterexample": synthetic,
        "original_handcrafted_solution_scope": {"modules": 1, "maximum_abs_weight": 6,
            "search_module_minimum": 4, "mutation_weight_bounds": [-3, 3]},
        "bounded_four_module_manual_solution": {"genome": genome.to_dict(), "episodes": len(episodes),
            "completed": successes, "optimal": optimal, "evolved": False,
            "scope": "within module-count and weight bounds; exact mutational reachability not established"},
        "extra_work": {"original_reproduction_episodes": 36000, "actual_act_diagnostic_episodes": 36000,
            "bounded_manual_solution_episodes": 1512, "new_searches": 0},
        "preservation": "original diagnostic, results, frozen sources and main experiment data not modified",
        "independent_scientific_review": False}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: output[k] for k in ("coordinate_first_step", "zero_logits_first_step", "tie_rule_counterexample")}, ensure_ascii=False), flush=True)
    print("범위 내 수작업 해:", successes, optimal, "/", len(episodes), flush=True)


if __name__ == "__main__":
    main()
