import tempfile
import unittest
from pathlib import Path

from evosim_interaction.analysis import create_summary, paired_decision
from evosim_interaction.search import run_search
from evosim_interaction.storage import read_json, write_json


class SearchTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_and_methods_share_initial_population(self):
        with tempfile.TemporaryDirectory() as temporary:
            full, resumed, control = [Path(temporary) / name for name in ("full", "resumed", "control")]
            for root in (full, resumed, control):
                write_json(root / "protocol.json", {"source_digest": "unit-test"})
            reference = run_search(full, 45, "evolution", rounds=2, size=4, pilot=True)
            run_search(resumed, 45, "evolution", rounds=2, size=4, pilot=True, pause_after=1)
            restarted = run_search(resumed, 45, "evolution", rounds=2, size=4, pilot=True)
            for key in ("best", "final", "initial", "history", "initial_population"):
                self.assertEqual(reference[key], restarted[key])
            for generation in range(2):
                self.assertEqual(read_json(full / "evolution_45" / f"round_{generation:03d}.json.gz"),
                                 read_json(resumed / "evolution_45" / f"round_{generation:03d}.json.gz"))
            random_result = run_search(control, 45, "random", rounds=2, size=4, pilot=True)
            self.assertEqual(reference["initial_population"], random_result["initial_population"])
            self.assertEqual(reference["validation_slots"], 0)
            self.assertEqual(random_result["validation_slots"], 0)

    def test_decision_rules_require_all_conditions_and_complete_seeds(self):
        passing = paired_decision([0.2] * 10)
        self.assertTrue(passing["passes_difference_rule"])
        self.assertEqual(passing["ci95"], [0.2, 0.2])
        self.assertFalse(paired_decision([0.01] * 10)["passes_difference_rule"])
        self.assertFalse(paired_decision([0.4, -0.4] * 5)["passes_difference_rule"])
        with self.assertRaises(ValueError):
            paired_decision([0.2] * 9)
        rows = [{"seed": seed, "evolution": {"success_rate": 0.8},
                 "random": {"success_rate": 0.3}, "probe": {"both_correct_rate": 0.0}}
                for seed in range(10)]
        summary = create_summary(rows, 0.1)
        self.assertFalse(summary["capability"]["q1_supported_by_declared_rule"])
        self.assertTrue(summary["evolution_advantage"]["passes_difference_rule"])


if __name__ == "__main__":
    unittest.main()
