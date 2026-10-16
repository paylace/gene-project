"""통계·감사·후보 잠금을 합성/훈련·검증 fixture에서 검사한다."""

import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from evosim_interaction_v02.analysis import sign_p, holm, primary_decision, diagnostic_summary
from evosim_interaction_v02.audit import audit_job
from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.evaluation import evaluate_policy
from evosim_interaction_v02.finalize import validate_blind_selection
from evosim_interaction_v02.network import Genome
from evosim_interaction_v02.policies import blind_patterns, FixedSequencePolicy
from evosim_interaction_v02.protocol import ROOT, prepare, load, evaluate_split
from evosim_interaction_v02.search import run_search, directory, lock_candidates, verify_candidate_lock
from evosim_interaction_v02.settings import Settings, digest_json
from evosim_interaction_v02.storage import read_json, write_json, file_digest

TEMP = ROOT / "tools/.verification_tmp"


class AnalysisTests(unittest.TestCase):
    def test_exact_sign_zero_eight_and_nine(self):
        self.assertEqual(sign_p(0), 1)
        self.assertEqual(sign_p(8), 0.0546875)
        self.assertEqual(sign_p(9), 0.0107421875)

    def test_holm_stepdown_ties_and_first_failure(self):
        self.assertEqual(holm([0.04, 0.045]), [False, False])
        self.assertEqual(holm([0.01, 0.04]), [True, True])
        self.assertEqual(holm([0.01, 0.01]), [True, True])
        self.assertEqual(holm([0.8, 0.01]), [False, True])

    def test_primary_selection_priority_and_mean_gate(self):
        result = primary_decision({"A0": [0] * 10, "A1": [0.2] * 10, "A2": [0.5] * 10})
        self.assertEqual(result["confirmation_condition"], "A1")
        result = primary_decision({"A0": [0] * 10, "A1": [0.01] * 10, "A2": [0.2] * 9 + [0]})
        self.assertEqual(result["confirmation_condition"], "A2")
        self.assertFalse(result["primary"]["A1"]["progress_gate"])
        result = primary_decision({"A0": [0] * 10, "A1": [0.2] * 8 + [0, 0], "A2": [0] * 10})
        self.assertIsNone(result["confirmation_condition"])

    def test_identical_conditions_reuse_bootstrap_and_zero_is_nonpositive(self):
        rates = {"A0": [0] * 10, "A1": [0.2] * 9 + [0], "A2": [0.2] * 9 + [0]}
        result = primary_decision(rates)
        self.assertEqual(result["primary"]["A1"], result["primary"]["A2"])
        self.assertEqual(result["primary"]["A1"]["positive_seeds"], 9)
        self.assertEqual(primary_decision({name: [0] * 10 for name in rates})["primary"]["A1"]["ci95"], [0, 0])
        with self.assertRaises(ValueError):
            primary_decision({"A0": [0] * 9, "A1": [0] * 10, "A2": [0] * 10})

    def test_diagnostics_missing_denominator_is_null(self):
        record = {"success": True, "actions": "212121", "failed_collect": 0, "walls": 0}
        value = diagnostic_summary([record])
        self.assertIsNone(value["repeated_fraction_failures"])
        self.assertIsNone(value["goal_change"]["fraction"])
        self.assertIsNone(value["mirror"]["both_correct_fraction"])


class AuditAndLockTests(unittest.TestCase):
    def temporary(self):
        return tempfile.TemporaryDirectory(dir=TEMP)

    def main_fixture(self, root, methods=None):
        # 정식 프로토콜 생성기는 60실행을 요구한다. 여기서는 잠금 경계용 작은 fixture만 만든다.
        config = prepare(root, "verification", verification_rounds=1, verification_size=2)
        config.update(phase="main", conditions=["A0"], methods=methods or ["evolution"], validation_rounds=[0])
        config["protocol_digest"] = digest_json({k: v for k, v in config.items() if k != "protocol_digest"})
        write_json(root / "protocol.json", config)
        return config

    def blind_fixture(self, root):
        patterns = blind_patterns()[:3]
        config = load(root)
        candidates = []
        for index, pattern in enumerate(patterns):
            metrics, records = evaluate_policy(FixedSequencePolicy(pattern), make_split().validation, 71002, "blind_validation")
            candidates.append({"index": index, "pattern": list(pattern), "metrics": metrics, "records": records})
        write_json(root / "blind_validation.json.gz", {"protocol_digest": config["protocol_digest"], "candidates": candidates})
        best = max(candidates, key=lambda c: (c["metrics"]["success_rate"], c["metrics"]["mean_reward"]))
        selection = {"index": best["index"], "pattern": best["pattern"], "validation_metrics": best["metrics"],
                     "validation_artifact_sha256": file_digest(root / "blind_validation.json.gz")}
        return patterns, selection

    def test_complete_genetic_audit_all_conditions_and_methods(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            prepare(root, "verification", verification_rounds=2)
            for condition in ("A0", "A1", "A2"):
                for method in ("evolution", "random"):
                    run_search(root, condition, method, 41)
                    result = audit_job(root, condition, method, 41)
                    self.assertEqual(result["genomes_checked"], 8)
                    self.assertEqual(result["episode_slots_checked"], 192)

    def test_audit_will_not_run_incomplete_search(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            prepare(root, "verification")
            run_search(root, "A0", "evolution", 41, pause_after=1)
            with self.assertRaises(ValueError):
                audit_job(root, "A0", "evolution", 41)
            self.assertFalse((directory(root, "A0", "evolution", 41) / "round_001.json.gz").exists())

    def test_lock_requires_all_jobs_and_test_has_no_kernel_call_before_lock(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            self.main_fixture(root, methods=["evolution", "random"])
            result = run_search(root, "A0", "evolution", 41)
            patterns, selection = self.blind_fixture(root)
            with patch("evosim_interaction_v02.finalize.blind_patterns", return_value=patterns):
                with self.assertRaises(ValueError):
                    lock_candidates(root, selection)
            genome = Genome.from_dict(result["best"]["genome"], Settings("A0"))
            with patch("evosim_interaction_v02.evaluation.evaluate") as kernel:
                with self.assertRaises(FileNotFoundError):
                    evaluate_split(root, genome, make_split().test[:1], 41, "test")
                kernel.assert_not_called()

    def test_locked_candidates_and_blind_tampering_are_rejected(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            self.main_fixture(root)
            result = run_search(root, "A0", "evolution", 41)
            patterns, selection = self.blind_fixture(root)
            with patch("evosim_interaction_v02.finalize.blind_patterns", return_value=patterns):
                validate_blind_selection(root, selection)
                lock_candidates(root, selection)
            verify_candidate_lock(root)
            genome = Genome.from_dict(result["best"]["genome"], Settings("A0"))
            # 경계만 검증하며 실제 held-out 성능은 계산하지 않는다.
            with patch("evosim_interaction_v02.evaluation.evaluate", return_value=({}, [])) as kernel:
                self.assertEqual(evaluate_split(root, genome, make_split().test[:1], 41, "test"), ({}, []))
                kernel.assert_called_once()
            write_json(root / "blind_validation.json.gz", {"tampered": True})
            with self.assertRaises(ValueError):
                verify_candidate_lock(root)

    def test_official_pilot_and_main_require_gate_before_creating_outputs(self):
        for phase in ("pilot", "main"):
            with self.assertRaises(ValueError):
                prepare(ROOT / "studies/interaction/results_v02" / phase, phase)

    def test_confirmation_requires_eligible_main_result(self):
        from evosim_interaction_v02.storage import manifest_digest, source_manifest
        source = manifest_digest(source_manifest())
        gate = {"passed": True, "source_digest": source}
        with self.temporary() as temporary:
            path = Path(temporary) / "gate.json"
            write_json(path, gate)
            from evosim_interaction_v02 import protocol
            original_read = protocol.read_json

            def no_eligible_condition(file):
                if Path(file).name == "summary.json":
                    return {"decision": {"confirmation_condition": None}}
                return original_read(file)

            with patch.object(protocol, "load", return_value={"source_digest": source}), patch.object(protocol, "read_json", side_effect=no_eligible_condition):
                with self.assertRaises(ValueError):
                    prepare(ROOT / "studies/interaction/results_v02/confirmation", "confirmation", gate=path)

    def test_summary_main_and_confirmation_use_registered_rules(self):
        from evosim_interaction_v02.analysis import summarize_results
        config = {"phase": "main", "seeds": list(range(10)), "conditions": ["A0", "A1", "A2"],
                  "methods": ["evolution", "random"], "protocol_digest": "unit-fixture"}

        def fake_read(path):
            path = Path(path)
            if path.name == "baseline_results.json":
                return {"blind": {"success_rate": 0.0}}
            if path.name == "probe.json.gz":
                return {"both_correct_rate": 1.0}
            condition = path.parent.parent.name
            rate = 0.0 if condition == "A0" or path.parent.name.startswith("random_") else 0.3
            return {"roles": {role: {"metrics": {"success_rate": rate}} for role in ("best", "initial", "final")}}

        with patch("evosim_interaction_v02.protocol.load", return_value=config), patch("evosim_interaction_v02.storage.read_json", side_effect=fake_read):
            summary = summarize_results(Path("fixture"))
            self.assertEqual(summary["decision"]["confirmation_condition"], "A1")
            self.assertTrue(summary["secondary_exploratory_unadjusted"]["A1"]["capability"]["capability_gate"])
            config["phase"], config["conditions"] = "confirmation", ["A0", "A1"]
            summary = summarize_results(Path("fixture"))
            self.assertTrue(summary["decision"]["confirmation"]["progress_gate"])


if __name__ == "__main__":
    unittest.main()
