"""사후 탐색 진단: 이동 또는 수집 한 부분만 정답 행동으로 보완한다."""

import argparse
import hashlib
from pathlib import Path

from evosim_interaction.datasets import make_split
from evosim_interaction.evaluation import evaluate_policy
from evosim_interaction.finalize import reconstruct_states
from evosim_interaction.network import Genome, NetworkPolicy
from evosim_interaction.storage import read_json, write_json


class AssistedPolicy:
    def __init__(self, genome, mode):
        self.network = NetworkPolicy(genome)
        self.mode = mode

    def reset(self):
        self.network.reset()

    def act(self, observation, rng):
        action = self.network.act(observation, rng)
        position, goal, _, _ = observation
        if self.mode == "collect_assisted" and position == goal:
            return 2
        if self.mode == "movement_assisted" and position != goal:
            return 1 if position < goal else 0
        return action


def probe(root):
    root = Path(root)
    output = root / "action_bottleneck"
    split = make_split()
    selections = {str(seed): read_json(root / f"evolution_{seed}" / "result.json")["best"]
                  for seed in range(201, 211)}
    protocol = {
        "status": "exploratory_after_primary_test", "retraining": False, "reselection": False,
        "seeds": list(range(201, 211)), "repeats": 3,
        "dataset": "same 504 primary test episodes; already observed",
        "conditions": {
            "collect_assisted": "at goal, force collect; otherwise original network action",
            "movement_assisted": "away from goal, force correct direction; at goal original network action",
        },
        "selected_digests": {seed: value["digest"] for seed, value in selections.items()},
        "network_state": "forward executes even when action is overridden",
        "interpretation": "diagnostic assistance only, not the unassisted ability of the individual",
        "script_digest": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if (output / "protocol.json").exists() and read_json(output / "protocol.json") != protocol:
        raise ValueError("Protocol changed")
    write_json(output / "protocol.json", protocol)
    rows = []
    for seed in range(201, 211):
        genome = Genome.from_dict(selections[str(seed)]["genome"])
        row = {"seed": seed}
        for condition in protocol["conditions"]:
            metrics, records = evaluate_policy(AssistedPolicy(genome, condition), split.test, seed, "test", repeats=3)
            reconstruct_states(split.test, records)
            write_json(output / f"{seed}_{condition}.json.gz", {"metrics": metrics, "records": records})
            row[condition] = metrics
        rows.append(row)
        print(f"ASSIST {seed}: collect={row['collect_assisted']['success_rate']:.4f} movement={row['movement_assisted']['success_rate']:.4f}", flush=True)
    summary = {"rows": rows, "means": {condition: sum(row[condition]["success_rate"] for row in rows) / len(rows)
                                        for condition in protocol["conditions"]}}
    write_json(output / "summary.json", summary)
    print(summary["means"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    probe(parser.parse_args().root)
