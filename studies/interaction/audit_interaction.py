"""실험 소스 밖에서 원자료·선발·예산을 감사하고 행동 실패를 요약한다."""

import argparse
import hashlib
import math
import zipfile
from collections import Counter
from pathlib import Path

from evosim_interaction.evaluation import selection_key, summarize
from evosim_interaction.network import Genome
from evosim_interaction.storage import read_json, write_json


def same_metrics(left, right):
    for key, value in right.items():
        if value is None:
            assert left[key] is None
        else:
            assert math.isclose(left[key], value, abs_tol=1e-12), (key, left[key], value)


def check_records(records):
    for row in records:
        assert len(row["actions"]) == len(row["rewards_cents"])
        assert set(row["actions"]) <= set("012")
        assert set(row["rewards_cents"]) <= {-1, 99}
        assert sum(reward == 99 for reward in row["rewards_cents"]) == row["collected"]
        assert row["success"] == (row["collected"] == 3)
        assert 1 <= len(row["actions"]) <= 24
        if not row["success"]:
            assert len(row["actions"]) == 24


def behavioral_counts(records):
    relation = {key: Counter() for key in ("left", "right", "at_goal")}
    collected_counts = Counter()
    first_changed_goal = Counter()
    repeated_action_failures = 0
    for record in records:
        collected_counts[record["collected"]] += 1
        if not record["success"] and len(set(record["actions"][-6:])) == 1:
            repeated_action_failures += 1
        for step, action in enumerate(record["actions"]):
            state = record["states"][step]
            label = "left" if state["goal"] < state["position"] else "right" if state["goal"] > state["position"] else "at_goal"
            ideal = {"left": "0", "right": "1", "at_goal": "2"}[label]
            relation[label]["steps"] += 1
            relation[label]["correct"] += action == ideal
            relation[label][f"action_{action}"] += 1
            if step and record["rewards_cents"][step - 1] == 99:
                first_changed_goal["steps"] += 1
                first_changed_goal["correct"] += action == ideal
    return {"relations": {key: dict(value) for key, value in relation.items()},
            "goals_collected": dict(collected_counts), "after_goal_change": dict(first_changed_goal),
            "failures_with_last_six_identical_actions": repeated_action_failures,
            "episodes": len(records)}


def audit(root):
    root = Path(root)
    protocol = read_json(root / "protocol.json")
    source_root = root.parent
    with zipfile.ZipFile(root / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for relative, expected in protocol["sources"].items():
            data = (source_root / relative).read_bytes()
            assert hashlib.sha256(data).hexdigest() == expected, relative
            archive.writestr(relative, data)
    counts = Counter()
    behavioral = {}
    for seed in protocol["seeds"]:
        initial_digests = []
        training_sets = {}
        for method in protocol["methods"]:
            directory = root / f"{method}_{seed}"
            result = read_json(directory / "result.json")
            initial_digests.append([Genome.from_dict(data).digest for data in result["initial_population"]])
            best = None
            seen_birth_ids = set()
            for generation in range(protocol["rounds"]):
                data = read_json(directory / f"round_{generation:03d}.json.gz")
                assert len(data["candidates"]) == 20
                if method == "evolution":
                    training_sets[generation] = data["training_episodes"]
                else:
                    assert training_sets[generation] == data["training_episodes"]
                for candidate in data["candidates"]:
                    genome = Genome.from_dict(candidate["genome"])
                    assert genome.digest == candidate["digest"]
                    birth = (candidate["ancestry"].get("mutation") or {}).get("added_module")
                    if birth:
                        assert birth not in seen_birth_ids
                        seen_birth_ids.add(birth)
                    for phase in ("training", "validation"):
                        if phase not in candidate:
                            continue
                        records = candidate[phase + "_records"]
                        check_records(records)
                        same_metrics(candidate[phase], summarize(records))
                        assert len(records) == (24 if phase == "training" else 252 * 3)
                        counts[phase + "_slots"] += 1
                        counts[phase + "_episodes"] += len(records)
                        counts[phase + "_actions"] += sum(len(r["actions"]) for r in records)
                    if "validation" in candidate:
                        item = (selection_key(candidate["validation"]), generation, candidate["slot"], candidate["digest"])
                        if best is None or item[0] > best[0]:
                            best = item
                del data
            assert (result["best"]["round"], result["best"]["slot"], result["best"]["digest"]) == best[1:]
            tested = read_json(directory / "test.json.gz")
            for role in ("best", "initial", "final"):
                check_records(tested[role]["records"])
                same_metrics(tested[role]["metrics"], summarize(tested[role]["records"]))
                assert len(tested[role]["records"]) == 504 * 3
                counts["test_candidate_episodes"] += len(tested[role]["records"])
            behavioral[f"{method}_{seed}"] = behavioral_counts(tested["best"]["records"])
            print(f"AUDIT {method} {seed} OK", flush=True)
        assert initial_digests[0] == initial_digests[1]
    assert counts["training_slots"] == 40000
    assert counts["validation_slots"] == 4400
    assert counts["training_episodes"] == 960000
    assert counts["validation_episodes"] == 3326400
    write_json(root / "audit.json", {"passed": True, "counts": dict(counts), "behavior": behavioral,
                                     "checks": ["source_hashes", "snapshot_archive", "all_training_and_validation_metrics",
                                                "selection_reconstruction", "matched_initial_populations",
                                                "matched_training_batches", "evaluation_budgets", "module_birth_ids",
                                                "test_metric_recomputation"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    audit(parser.parse_args().root)
