"""실험을 실행하지 않고 계획의 예산·격리·변경 거부를 검사한다."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.q1_plan import DEFAULT, ROOT, local_path, preview


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(DEFAULT.read_text(encoding="utf-8"))

    def test_budget_and_no_execution(self):
        result = preview(self.config)
        self.assertFalse(result["execution_ready"])
        self.assertEqual(result["experiments_executed"], 0)
        self.assertEqual(result["phases"]["pilot"]["training_episodes"], 28800)
        self.assertEqual(result["phases"]["main"]["validation_episodes"], 9979200)
        self.assertEqual(result["phases"]["main"]["test_role_episodes"], 272160)
        self.assertEqual(result["total_planned_episodes_including_conditional_confirmation_and_shared_baselines"], 21960768)
        self.assertEqual(result["auxiliary_budget"]["main_counterfactual_extra_forward_calls_upper_bound"], 4354560)

    def test_job_identity_unique_and_confirmation_conditional(self):
        result = preview(self.config)
        jobs = result["jobs"]
        self.assertEqual(len(jobs), 112)
        self.assertEqual(len({j["planned_directory"] for j in jobs}), 112)
        self.assertEqual(sum(j["conditional"] for j in jobs), 40)
        self.assertEqual(result["phases"]["main"]["probe_pair_trials"], 6300)
        self.assertEqual(result["phases"]["confirmation"]["probe_forward_calls"], 8400)

    def test_original_weights_are_not_probabilities(self):
        result = preview(self.config)
        self.assertAlmostEqual(result["mutation_weight_sum"], 1.10)
        self.assertAlmostEqual(sum(result["mutation_probabilities"]), 1.0)
        self.assertAlmostEqual(result["mutation_probabilities"][4], 0.2)

    def test_split_lists_and_auxiliary_counts(self):
        result = preview(self.config)
        datasets = result["datasets"]
        self.assertEqual([len(datasets[k]) for k in ("train", "validation", "test")], [756, 252, 504])
        self.assertEqual(len(set().union(*map(set, datasets.values()))), 1512)
        self.assertEqual(len(result["blind_patterns"]), 140)
        self.assertEqual(len(result["probe_pairs"]), 35)
        self.assertEqual(result["auxiliary_budget"]["shared_blind_validation_episodes"], 35280)
        self.assertEqual(result["auxiliary_budget"]["shared_baseline_test_episodes"], 11088)

    def test_draft_cannot_claim_locked_or_implemented(self):
        for key, value in (("locked", True), ("status", "ready"), ("implementation_source_digest", "abc")):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                preview(config)

    def test_pilot_test_and_validation_rejected(self):
        for key in ("test", "validation"):
            config = copy.deepcopy(self.config)
            config["phases"]["pilot"][key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                preview(config)

    def test_seed_overlap_and_duplicates_rejected(self):
        for replacement in (self.config["phases"]["main"]["seeds"][:], [301] * 10):
            self.config["phases"]["confirmation"]["seeds"] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                preview(self.config)

    def test_condition_and_operator_confound_rejected(self):
        self.config["conditions"]["A1"]["other_recurrence"] = False
        with self.assertRaises(ValueError):
            preview(self.config)
        self.setUp()
        self.config["operators"]["forbidden_proposal"] = "resample_until_valid"
        with self.assertRaises(ValueError):
            preview(self.config)

    def test_budget_and_selection_drift_rejected(self):
        self.config["phases"]["main"]["expected_train_validation"] -= 1
        with self.assertRaises(ValueError):
            preview(self.config)
        self.setUp()
        self.config["search"]["validation_rounds"].remove(0)
        with self.assertRaises(ValueError):
            preview(self.config)

    def test_dataset_and_statistical_drift_rejected(self):
        self.config["dataset"]["seed"] += 1
        with self.assertRaises(ValueError):
            preview(self.config)
        self.setUp()
        self.config["analysis"]["alpha"] = 0.10
        with self.assertRaises(ValueError):
            preview(self.config)

    def test_source_drift_rejected_without_editing_frozen_files(self):
        read_bytes = Path.read_bytes
        target = ROOT / "studies/interaction/evosim_interaction/network.py"

        def changed(path):
            content = read_bytes(path)
            return content + b"# simulated drift" if path == target else content

        with patch.object(Path, "read_bytes", changed), self.assertRaisesRegex(ValueError, "동결 소스 변경"):
            preview(self.config)

    def test_unsafe_or_old_output_paths_rejected(self):
        with self.assertRaises(ValueError):
            local_path("../outside")
        self.config["paths"]["results"] = "studies/interaction/evosim_interaction/results_v01"
        with self.assertRaises(ValueError):
            preview(self.config)


if __name__ == "__main__":
    unittest.main()
