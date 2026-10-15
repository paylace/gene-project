"""본 실험 이후의 탐색적 원인 진단. 후보 재선발이나 재훈련을 하지 않는다."""

import argparse
import hashlib
from pathlib import Path

from evosim_interaction.datasets import all_episodes, make_split
from evosim_interaction.evaluation import evaluate_policy
from evosim_interaction.finalize import reconstruct_states
from evosim_interaction.network import Genome, Module, NetworkPolicy
from evosim_interaction.storage import read_json, write_json


class NoInternalStateModule(Module):
    def compute(self, weighted, previous):
        # Keep EMA amplitude and latch drive threshold; remove only prior module state.
        return super().compute(weighted, 0.0)


def witness_genome():
    return Genome({
        "m0": Module("threshold", -1 / 12, 0.5),
        "m1": Module("threshold", -1 / 12, 0.5),
        "m2": Module("threshold", 0.1, 0.5),
        "m3": Module("tanh", 0, 0.5),
    }, {
        "m0": {"in_0": 1.0, "in_1": -1.0},
        "m1": {"in_0": -1.0, "in_1": 1.0},
        "out_0": {"m0": 1.0}, "out_1": {"m1": 1.0}, "out_2": {"m2": 0.5},
    }, {})


def diagnose(root):
    root = Path(root)
    output = root / "exploratory_diagnostics"
    split = make_split()
    conditions = ("graph_off", "internal_state_off", "all_temporal_off")
    selections = {str(seed): read_json(root / f"evolution_{seed}" / "result.json")["best"]
                  for seed in range(201, 211)}
    protocol = {
        "status": "exploratory_after_primary_test", "retraining": False, "reselection": False,
        "dataset": "same 504 held-out episodes, already used in primary evaluation",
        "seeds": list(range(201, 211)), "conditions": list(conditions), "repeats": 3,
        "selected_digests": {seed: item["digest"] for seed, item in selections.items()},
        "witness": "hand constructed four-module threshold comparator; not an evolved individual",
        "script_digest": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if (output / "protocol.json").exists() and read_json(output / "protocol.json") != protocol:
        raise ValueError("Diagnostic protocol changed")
    write_json(output / "protocol.json", protocol)
    witness = witness_genome()
    metrics, records = evaluate_policy(NetworkPolicy(witness), all_episodes(), 71005, "witness")
    if metrics["success_rate"] != 1.0:
        raise AssertionError("Hand constructed witness failed")
    if any(len(row["actions"]) != row["optimal_steps"] for row in records):
        raise AssertionError("Witness was not optimal")
    write_json(output / "witness.json.gz", {"genome": witness.to_dict(), "metrics": metrics, "records": records})
    summaries = []
    for seed in range(201, 211):
        baseline = read_json(root / f"evolution_{seed}" / "test.json.gz")["best"]
        original = Genome.from_dict(selections[str(seed)]["genome"])
        # All primary selected policies are also checked through the independent scalar path.
        scalar_metrics, scalar_records = evaluate_policy(NetworkPolicy(original), split.test, seed, "test", repeats=3)
        compact_baseline = [{k: v for k, v in row.items() if k != "states"} for row in baseline["records"]]
        if scalar_records != compact_baseline:
            raise AssertionError(f"Primary batch/scalar mismatch at seed {seed}")
        row = {"seed": seed, "baseline": baseline["metrics"], "scalar_matches_primary": True}
        for condition in conditions:
            genome = original.clone()
            if condition in ("graph_off", "all_temporal_off"):
                genome.rec = {}
            if condition in ("internal_state_off", "all_temporal_off"):
                genome.modules = {key: NoInternalStateModule(mod.op, mod.bias, mod.memory_decay)
                                  for key, mod in genome.modules.items()}
            result_metrics, result_records = evaluate_policy(NetworkPolicy(genome), split.test, seed, "test", repeats=3)
            reconstruct_states(split.test, result_records)
            write_json(output / f"{seed}_{condition}.json.gz", {"metrics": result_metrics, "records": result_records})
            row[condition] = result_metrics
        summaries.append(row)
        print(f"DIAG {seed}: " + ", ".join(f"{condition}={row[condition]['success_rate']:.4f}" for condition in conditions), flush=True)
    result = {"witness": metrics, "rows": summaries,
              "means": {condition: sum(row[condition]["success_rate"] for row in summaries) / 10
                        for condition in ("baseline",) + conditions}}
    write_json(output / "summary.json", result)
    print(result["means"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    diagnose(parser.parse_args().root)
