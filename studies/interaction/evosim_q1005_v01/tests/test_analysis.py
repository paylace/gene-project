import math
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from evosim_interaction_v02.storage import write_json
from evosim_q1005_v01.analysis import sign_p, paired_summary, primary_decision, summarize
from evosim_q1005_v01.protocol import jobs_for
from evosim_q1005_v01.source import ROOT


class AnalysisTests(unittest.TestCase):
    def test_exact_ten_percentage_point_boundary_is_not_lost_to_float_subtraction(self):
        decision = primary_decision({"R0S0": [0.2] * 20, "R1S0": [0.3] * 20,
                                     "R0S1": [0.2] * 20, "R1S1": [0.2] * 20})
        self.assertTrue(decision["primary"]["R1S0"]["progress_gate"])
        below = primary_decision({"R0S0": [0.2] * 20, "R1S0": [0.29999] * 20,
                                  "R0S1": [0.2] * 20, "R1S1": [0.2] * 20})
        self.assertFalse(below["primary"]["R1S0"]["progress_gate"])

    def test_exact_twenty_seed_sign_probabilities(self):
        self.assertEqual(sign_p(20), 1 / 2 ** 20)
        self.assertAlmostEqual(sign_p(15), 0.020694732666015625)
        self.assertEqual(sign_p(0), 1)
        with self.assertRaises(ValueError):
            sign_p(21)

    def test_all_positive_treatments_use_registered_priority_not_largest_mean(self):
        decision = primary_decision({"R0S0": [0.1] * 20, "R1S0": [0.3] * 20,
                                     "R0S1": [0.5] * 20, "R1S1": [0.9] * 20})
        self.assertEqual(decision["confirmation_condition"], "R1S0")
        self.assertTrue(all(r["holm_pass"] and r["progress_gate"] for r in decision["primary"].values()))

    def test_combination_can_be_confirmed_only_when_registered_single_factors_fail(self):
        decision = primary_decision({"R0S0": [0] * 20, "R1S0": [0] * 20,
                                     "R0S1": [0] * 20, "R1S1": [0.8] * 20})
        self.assertEqual(decision["confirmation_condition"], "R1S1")
        self.assertAlmostEqual(decision["interaction_exploratory"]["mean"], 0.8)

    def test_small_mean_and_zero_ties_fail_progress(self):
        decision = primary_decision({"R0S0": [0.1] * 20, "R1S0": [0.15] * 20,
                                     "R0S1": [0.1] * 20, "R1S1": [0.1] * 20})
        self.assertTrue(decision["primary"]["R1S0"]["holm_pass"])
        self.assertFalse(decision["primary"]["R1S0"]["progress_gate"])
        self.assertEqual(decision["primary"]["R0S1"]["positive_seeds"], 0)
        self.assertIsNone(decision["confirmation_condition"])

    def test_holm_three_contrasts_rejects_fifteen_positive_when_first_threshold_fails(self):
        values = [0.8] * 15 + [0] * 5
        decision = primary_decision({"R0S0": [0] * 20, "R1S0": values, "R0S1": values, "R1S1": values})
        self.assertTrue(all(r["positive_seeds"] == 15 for r in decision["primary"].values()))
        self.assertTrue(all(not r["holm_pass"] for r in decision["primary"].values()))
        self.assertIsNone(decision["confirmation_condition"])

    def test_invalid_missing_and_nonfinite_cohorts_are_rejected(self):
        valid = {c: [0.0] * 20 for c in ("R0S0", "R1S0", "R0S1", "R1S1")}
        for replacement in ([0] * 19, [float("nan")] * 20, [1.1] * 20):
            with self.assertRaises(ValueError):
                primary_decision({**valid, "R1S0": replacement})
        with self.assertRaises(ValueError):
            primary_decision({"R0S0": [0] * 20})

    def test_pair_summary_median_and_intervals_use_seed_unit(self):
        draws = np.random.default_rng(72005).integers(0, 20, size=(10000, 20))
        row = paired_summary([0.2] * 20, draws)
        self.assertAlmostEqual(row["mean"], 0.2)
        self.assertEqual(row["median"], 0.2)
        self.assertTrue(all(abs(v - 0.2) < 1e-10 for v in row["ci95"]))

    def test_main_and_confirmation_summaries_share_only_one_random_control(self):
        temporary = ROOT / "tools/.verification_tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as folder:
            root = Path(folder)
            for phase, conditions, seeds in (("main", ["R0S0", "R1S0", "R0S1", "R1S1"], list(range(401, 421))),
                                             ("confirmation", ["R0S0", "R1S0"], list(range(501, 521)))):
                config = {"phase": phase, "conditions": conditions, "seeds": seeds, "jobs": jobs_for(conditions, seeds),
                          "protocol_digest": "synthetic_fixture"}
                for job in config["jobs"]:
                    path = root / job["condition"] / f"{job['method']}_{job['seed']}"
                    rate = 0.1 if job["method"] == "random" or job["condition"] == "R0S0" else 0.5
                    write_json(path / "test.json.gz", {"roles": {role: {"metrics": {"success_rate": rate}}
                                                                 for role in ("best", "initial", "final")}})
                    write_json(path / "probe.json.gz", {"both_correct_rate": 0.2})
                    write_json(path / "diagnostics.json", {"fixture": True})
                write_json(root / "baseline_results.json", {"blind": {"success_rate": 0.05}})
                with patch("evosim_q1005_v01.analysis.load", return_value=config):
                    result = summarize(root)
                self.assertTrue(result["random_control_shared"])
                self.assertEqual(len(result["rates"]["R0S0"]["random"]), 20)
                if phase == "main":
                    self.assertEqual(result["decision"]["confirmation_condition"], "R1S0")
                else:
                    self.assertTrue(result["decision"]["confirmation"]["progress_gate"])


if __name__ == "__main__":
    unittest.main()
