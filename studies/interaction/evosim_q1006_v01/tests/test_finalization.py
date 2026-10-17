import copy
import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.evaluation import evaluate_policy
from evosim_q1006_v01.network import initial_genome
from evosim_interaction_v02.policies import CoordinatePolicy, FixedSequencePolicy
from evosim_interaction_v02.settings import digest_json
from evosim_interaction_v02.storage import read_json, write_json, file_digest
from evosim_q1006_v01.model import NETWORK, Treatment
from evosim_q1006_v01.source import ROOT
from evosim_q1006_v01.finalize import evaluate_locked, validate_blind
from evosim_q1006_v01.audit import audit_test_job
from evosim_q1006_v01.protocol import evaluate_split
from evosim_q1006_v01.search import verify_candidate_lock


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        temporary = ROOT / "tools/.verification_tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_mocked_test_pipeline_caches_three_roles_and_resumes_without_new_policy_test(self):
        # 후보 held-out 성능은 mock한다. 알려진 좌표 규칙 기록은 저장·재생 검사 용도다.
        job = {"condition": "F0", "method": "evolution", "seed": 41}
        genome = initial_genome(random.Random(5), NETWORK)
        candidate = {"genome": genome.to_dict(), "digest": genome.digest}
        path = self.root / "F0/evolution_41"
        write_json(path / "result.json", {r: candidate for r in ("best", "initial", "final")})
        config = {"phase": "confirmation", "protocol_digest": "mock_fixture", "jobs": [job]}
        locked = {"lock_digest": "mock_lock", "jobs": {"F0/evolution_41": {
            "roles": {r: {"digest": genome.digest} for r in ("best", "initial", "final")}}}}
        metrics, records = evaluate_policy(CoordinatePolicy(), make_split().test, 1, "coordinate_storage_fixture", repeats=3)
        write_json(self.root / "baseline_results.json", {"fixture": True})
        with (patch("evosim_q1006_v01.finalize.load", return_value=config),
              patch("evosim_q1006_v01.finalize.verify_candidate_lock", return_value=locked),
              patch("evosim_q1006_v01.finalize.evaluate_split", return_value=(metrics, records)) as calculate,
              patch("evosim_q1006_v01.finalize.audit_test_job", return_value={"passed": True}),
              patch("evosim_q1006_v01.finalize.summarize", return_value={"fixture": True})):
            evaluate_locked(self.root)
            evaluate_locked(self.root)
            self.assertEqual(calculate.call_count, 1)
        tested = read_json(path / "test.json.gz")
        self.assertEqual(tested["unique_role_episode_calls"], 1512)
        self.assertEqual(tested["actual_episode_accounting"]["known_completed_episode_calls"], 1512)
        with (patch("evosim_q1006_v01.audit.load", return_value=config),
              patch("evosim_q1006_v01.audit.verify_candidate_lock", return_value=locked)):
            self.assertTrue(audit_test_job(self.root, job)["passed"])
            cache_path = path / "role_cache" / (genome.digest + ".json.gz")
            changed = read_json(cache_path)
            changed["payload"]["metrics"]["success_rate"] = 0
            write_json(cache_path, changed)
            with self.assertRaises(ValueError):
                audit_test_job(self.root, job)

    def test_test_kernel_not_called_without_lock_or_for_another_genome_or_partial_test(self):
        job = {"condition": "F1", "method": "evolution", "seed": 41}
        genome = initial_genome(random.Random(5), NETWORK)
        config = {"phase": "main", "jobs": [job], "validation_rounds": [0]}
        with (patch("evosim_q1006_v01.protocol.load", return_value=config),
              patch("evosim_q1006_v01.evaluation.evaluate") as calculate):
            with patch("evosim_q1006_v01.search.verify_candidate_lock", side_effect=ValueError("missing lock")):
                with self.assertRaises(ValueError):
                    evaluate_split(self.root, **job, genome=genome, episodes=make_split().test, purpose="test", repeats=3)
            wrong = {"jobs": {"F1/evolution_41": {"roles": {"best": {"digest": "wrong"}}}}}
            with patch("evosim_q1006_v01.search.verify_candidate_lock", return_value=wrong):
                with self.assertRaises(ValueError):
                    evaluate_split(self.root, **job, genome=genome, episodes=make_split().test, purpose="test", repeats=3)
            matching = {"jobs": {"F1/evolution_41": {"roles": {"best": {"digest": genome.digest}}}}}
            with patch("evosim_q1006_v01.search.verify_candidate_lock", return_value=matching):
                with self.assertRaises(ValueError):
                    evaluate_split(self.root, **job, genome=genome, episodes=make_split().test[:1], purpose="test", repeats=3)
            calculate.assert_not_called()

    def test_blind_first_tie_and_changed_validation_artifact_rejected(self):
        from types import SimpleNamespace
        # 작은 훈련 목록에서 고정 순서의 선발·잠금 논리만 검사한다.
        episodes = make_split().train[:2]
        patterns = [(0,), (1,)]
        candidates = []
        for index, pattern in enumerate(patterns):
            metrics, records = evaluate_policy(FixedSequencePolicy(pattern), episodes, 1, "blind_fixture")
            candidates.append({"index": index, "pattern": list(pattern), "metrics": metrics, "records": records})
        artifact = self.root / "blind_validation.json.gz"
        write_json(artifact, {"candidates": candidates})
        selected = {"index": 0, "pattern": [0], "validation_metrics": candidates[0]["metrics"],
                    "validation_artifact_sha256": file_digest(artifact)}
        with (patch("evosim_q1006_v01.finalize.load", return_value={"phase": "main"}),
              patch("evosim_q1006_v01.finalize.make_split", return_value=SimpleNamespace(validation=episodes)),
              patch("evosim_q1006_v01.finalize.blind_patterns", return_value=patterns)):
            self.assertEqual(validate_blind(self.root, selected), selected)
            second = {**selected, "index": 1, "pattern": [1]}
            with self.assertRaises(ValueError):
                validate_blind(self.root, second)
        config = {"phase": "main", "protocol_digest": "fixture"}
        manifest = {"protocol_digest": "fixture", "jobs": {}}
        locked = {**manifest, "blind": selected}
        locked["lock_digest"] = digest_json(locked)
        write_json(self.root / "candidate_lock.json", locked)
        with patch("evosim_q1006_v01.search.candidate_manifest", return_value=manifest):
            self.assertEqual(verify_candidate_lock(self.root, config), locked)
            write_json(artifact, {"candidates": []})
            with self.assertRaises(ValueError):
                verify_candidate_lock(self.root, config)


if __name__ == "__main__":
    unittest.main()
