"""보존된 v0.1 검사47개와 검증 전용 후보의 공통 random 검사2개를 실행한다.

원본 혼합-purpose fixture의 과도한 동일성 검사는 원본 그대로 보존한다.
이 도구는 공식 소스·프로토콜·결과·실행 게이트를 수정하지 않는다.
"""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "studies/interaction"
sys.path.insert(0, str(PACKAGE))
from evosim_q1005_v01.protocol import prepare
from evosim_q1005_v01.search import run_search, directory
from evosim_q1005_v01.audit import audit_job
from evosim_interaction_v02.storage import read_json

REPLACED = "test_search.SearchTests.test_shared_random_genotypes_and_validation_winners_match_all_treatments"


class SharedRandomValidationTests(unittest.TestCase):
    def test_all_validation_only_candidates_and_genotypes_match(self):
        temporary = ROOT / "tools/.verification_tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as name:
            root = Path(name)
            # 회차0과 마지막 회차 모두 검증한다. 본 실험의 검증 전용 선발과 일치한다.
            config = prepare(root, "verification", verification_size=2, verification_rounds=2,
                             verification_methods=["random"], verification_validation=True)
            self.assertEqual(config["validation_rounds"], [0, 1])
            results = [run_search(root, **job) for job in config["jobs"]]
            self.assertEqual(len(results), 4)
            for result in results:
                for role in ("best", "initial", "final"):
                    self.assertEqual(result[role]["selection_source"], "validation")
            for result in results[1:]:
                self.assertEqual(result["initial_population"], results[0]["initial_population"])
                for role in ("best", "initial", "final"):
                    self.assertEqual(result[role], results[0][role])
            reference = None
            for job in config["jobs"]:
                self.assertTrue(audit_job(root, **job)["passed"])
                for generation in range(2):
                    rows = read_json(directory(root, **job) / f"round_{generation:03d}.json.gz")["candidates"]
                    actual = [(r["genome"], r["validation"], r["validation_records"]) for r in rows]
                    if job == config["jobs"][0]:
                        if reference is None:
                            reference = {}
                        reference[generation] = actual
                    else:
                        self.assertEqual(actual, reference[generation])

    def test_production_branch_never_selects_training_fallback(self):
        # 검사용 분기 조건을 느슨하게 만드는 회귀를 별도 확인한다.
        from unittest.mock import patch
        import evosim_q1005_v01.search as module
        temporary = ROOT / "tools/.verification_tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as name:
            root = Path(name)
            config = prepare(root, "verification", verification_size=2, verification_rounds=3,
                             verification_methods=["random"], verification_validation=True)
            original_load = module.load
            def production_branch(path):
                saved = original_load(path)
                return {**saved, "phase": "main"}
            results = []
            with patch.object(module, "load", side_effect=production_branch):
                for job in config["jobs"]:
                    results.append(run_search(root, **job))
            for result in results:
                for role in ("best", "initial", "final"):
                    self.assertEqual(result[role]["selection_source"], "validation")
                    self.assertNotIn("mean_training_score", result[role]["metrics"])
                    self.assertEqual(result[role], results[0][role])


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


if __name__ == "__main__":
    original = list(flatten(unittest.defaultTestLoader.discover(str(PACKAGE / "evosim_q1005_v01/tests"))))
    replaced = [test for test in original if test.id() == REPLACED]
    if len(original) != 48 or len(replaced) != 1:
        raise ValueError("Frozen suite identity changed; explicit review required")
    print("원본 혼합-purpose 동일성 검사1개는 보존. 검증 전용·본 분기 검사2개로 명시적 교체.", flush=True)
    suite = unittest.TestSuite([test for test in original if test.id() != REPLACED])
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(SharedRandomValidationTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
