import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from evosim_interaction_v02.protocol import prepare as old_prepare
from evosim_interaction_v02.search import run_search as old_search
from evosim_interaction_v02.settings import digest_json
from evosim_interaction_v02.storage import read_json, write_json, file_digest, manifest_digest
from evosim_q1005_v01.model import CONDITIONS
from evosim_q1005_v01.protocol import prepare, load, RESULTS, jobs_for
from evosim_q1005_v01.search import run_search, directory, attempt_accounting, candidate_manifest, lock_candidates
from evosim_q1005_v01.audit import audit_job
from evosim_q1005_v01.finalize import cached_evaluation
from evosim_q1005_v01.source import ROOT, source_manifest, plan


class SearchTests(unittest.TestCase):
    def setUp(self):
        temporary = ROOT / "tools/.verification_tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_pause_resume_all_four_conditions_and_both_methods(self):
        a, b = self.root / "continuous", self.root / "resumed"
        ca = prepare(a, "verification", verification_size=2, verification_rounds=3)
        prepare(b, "verification", verification_size=2, verification_rounds=3)
        for job in ca["jobs"]:
            direct = run_search(a, **job)
            run_search(b, **job, pause_after=1)
            before = file_digest(directory(b, **job) / "round_000.json.gz")
            resumed = run_search(b, **job)
            self.assertEqual(direct, resumed)
            self.assertEqual(before, file_digest(directory(b, **job) / "round_000.json.gz"))
            for key in ("population", "ancestry", "rng_state", "next_module_id", "history", "best", "initial", "record_hashes"):
                self.assertEqual(read_json(directory(a, **job) / "checkpoint.json")[key],
                                 read_json(directory(b, **job) / "checkpoint.json")[key])
            self.assertTrue(audit_job(b, **job)["passed"])

    def test_r0s0_search_genomes_rng_and_winners_match_old_a0(self):
        a, b = self.root / "old", self.root / "new"
        old_prepare(a, "verification", verification_size=4, verification_rounds=5)
        prepare(b, "verification", verification_conditions=["R0S0"], verification_methods=["evolution"],
                verification_size=4, verification_rounds=5)
        old = old_search(a, "A0", "evolution", 41)
        new = run_search(b, "R0S0", "evolution", 41)
        self.assertEqual(old["initial_population"], new["initial_population"])
        for role in ("best", "initial", "final"):
            for key in ("genome", "digest", "slot", "round"):
                self.assertEqual(old[role][key], new[role][key])
            for key, value in old[role]["metrics"].items():
                self.assertEqual(new[role]["metrics"][key], value)
        oa = read_json(a / "A0/evolution_41/checkpoint.json")
        nb = read_json(b / "R0S0/evolution_41/checkpoint.json")
        self.assertEqual(oa["rng_state"], nb["rng_state"])
        self.assertEqual(oa["population"], nb["population"])

    def test_shared_random_genotypes_and_validation_winners_match_all_treatments(self):
        config = prepare(self.root, "verification", verification_size=2, verification_rounds=3,
                         verification_methods=["random"], verification_validation=True)
        results = [run_search(self.root, **job) for job in config["jobs"]]
        for result in results[1:]:
            self.assertEqual(result["initial_population"], results[0]["initial_population"])
            for role in ("best", "initial", "final"):
                self.assertEqual(result[role], results[0][role])
        for job in config["jobs"]:
            self.assertTrue(audit_job(self.root, **job)["passed"])

    def test_orphan_round_is_preserved_and_recalculation_counted(self):
        config = prepare(self.root, "verification", verification_conditions=["R1S1"],
                         verification_methods=["evolution"], verification_size=2, verification_rounds=3)
        job = config["jobs"][0]
        run_search(self.root, **job, pause_after=1)
        import evosim_q1005_v01.search as module
        original = module.write_json
        def fail_checkpoint(path, value):
            if Path(path).name == "checkpoint.json":
                raise RuntimeError("injected after immutable round write")
            return original(path, value)
        with patch.object(module, "write_json", side_effect=fail_checkpoint):
            with self.assertRaises(RuntimeError):
                run_search(self.root, **job)
        path = directory(self.root, **job)
        digest = file_digest(path / "round_001.json.gz")
        run_search(self.root, **job)
        self.assertEqual(digest, file_digest(path / "round_001.json.gz"))
        counts = attempt_accounting(path)
        self.assertEqual(counts["known_completed_episode_calls"], 192)
        self.assertEqual(counts["interrupted_attempts"], 0)

    def test_checkpoint_round_result_and_cross_condition_tampering_rejected(self):
        config = prepare(self.root, "verification", verification_conditions=["R0S0"],
                         verification_methods=["evolution"], verification_size=2, verification_rounds=2)
        job = config["jobs"][0]
        run_search(self.root, **job)
        path = directory(self.root, **job)
        cp = read_json(path / "checkpoint.json")
        altered = copy.deepcopy(cp)
        altered["next_module_id"] += 1
        write_json(path / "checkpoint.json", altered)
        with self.assertRaises(ValueError):
            run_search(self.root, **job)
        write_json(path / "checkpoint.json", cp)
        result = read_json(path / "result.json")
        altered = copy.deepcopy(result)
        altered["identity"]["condition"] = "R1S1"
        write_json(path / "result.json", altered)
        with self.assertRaises(ValueError):
            run_search(self.root, **job)
        write_json(path / "result.json", result)
        saved = read_json(path / "round_000.json.gz")
        saved["extra"] = 1
        write_json(path / "round_000.json.gz", saved)
        with self.assertRaises(ValueError):
            run_search(self.root, **job)

    def test_complete_genetic_score_and_trajectory_audit_rejects_corruption(self):
        config = prepare(self.root, "verification", verification_conditions=["R1S1"],
                         verification_methods=["evolution"], verification_size=2, verification_rounds=3)
        job = config["jobs"][0]
        run_search(self.root, **job)
        self.assertTrue(audit_job(self.root, **job)["passed"])
        path = directory(self.root, **job)
        saved = read_json(path / "round_000.json.gz")
        saved["ranking_diagnostics"]["completion_score_conflict_pairs"] += 1
        write_json(path / "round_000.json.gz", saved)
        # 원자료 해시 자체가 먼저 변조를 잡는다.
        with self.assertRaises(ValueError):
            audit_job(self.root, **job)

    def test_every_job_must_finish_and_pilot_cannot_lock(self):
        config = prepare(self.root, "verification", verification_size=2, verification_rounds=1)
        run_search(self.root, **config["jobs"][0])
        with self.assertRaises(ValueError):
            candidate_manifest(self.root, config)
        with self.assertRaises(ValueError):
            lock_candidates(self.root, {})

    def test_protocol_source_dataset_and_design_changes_rejected(self):
        prepare(self.root, "verification", verification_size=2, verification_rounds=1)
        current = source_manifest()
        with patch("evosim_q1005_v01.protocol.source_manifest", return_value={**current, "changed": "x"}):
            with self.assertRaises(ValueError):
                load(self.root)
        original = read_json(self.root / "datasets.json")
        changed = copy.deepcopy(original)
        changed["train"].pop()
        write_json(self.root / "datasets.json", changed)
        with self.assertRaises(ValueError):
            load(self.root)
        write_json(self.root / "datasets.json", original)
        spec = plan()
        spec["reward"]["eta"] = 0.2
        with patch("evosim_q1005_v01.protocol.plan", return_value=spec):
            with self.assertRaises(ValueError):
                load(self.root)

    def test_production_prepare_requires_matching_gate_pilot_and_canonical_path(self):
        with self.assertRaises(ValueError):
            prepare(self.root / "main", "main")
        sources = source_manifest()
        gate = self.root / "gate.json"
        write_json(gate, {"passed": True, "full_gate": True, "source_digest": manifest_digest(sources), "plan_digest": digest_json(plan())})
        with patch("evosim_q1005_v01.protocol.RESULTS", self.root):
            with self.assertRaises(ValueError):
                prepare(self.root / "main", "main", gate=gate)
            bad_pilot = self.root / "bad_pilot.json"
            write_json(bad_pilot, {"all_jobs_complete": True, "source_digest": "wrong", "plan_digest": digest_json(plan())})
            with self.assertRaises(ValueError):
                prepare(self.root / "main", "main", gate=gate, pilot_report=bad_pilot)
            write_json(bad_pilot, {"all_jobs_complete": True, "source_digest": manifest_digest(sources), "plan_digest": digest_json(plan())})
            config = prepare(self.root / "main", "main", gate=gate, pilot_report=bad_pilot)
            self.assertEqual(len(config["jobs"]), 100)
            self.assertEqual(sum(j["method"] == "random" for j in config["jobs"]), 20)
            self.assertEqual(load(self.root / "main"), config)


class WorkLedgerTests(unittest.TestCase):
    def setUp(self):
        temporary = ROOT / "tools/.verification_tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_cache_reuses_work_and_rejects_changed_identity_payload(self):
        path = self.root / "cache.json.gz"
        identity = {"protocol": "fixed"}
        calls = []
        def calculate():
            calls.append(1)
            return {"metric": 0.4}, 3
        a = cached_evaluation(self.root, path, identity, 3, calculate)
        b = cached_evaluation(self.root, path, identity, 3, calculate)
        self.assertEqual(a, b)
        self.assertEqual(len(calls), 1)
        self.assertEqual(attempt_accounting(self.root)["known_completed_episode_calls"], 3)
        with self.assertRaises(ValueError):
            cached_evaluation(self.root, path, {"protocol": "changed"}, 3, calculate)
        wrapper = read_json(path)
        wrapper["payload"]["metric"] = 9
        write_json(path, wrapper)
        with self.assertRaises(ValueError):
            cached_evaluation(self.root, path, identity, 3, calculate)

    def test_unknown_interrupted_episode_work_keeps_upper_bound(self):
        path = self.root / "cache.json.gz"
        def fail():
            raise RuntimeError("injected inside evaluation")
        with self.assertRaises(RuntimeError):
            cached_evaluation(self.root, path, {}, 3, fail)
        cached_evaluation(self.root, path, {}, 3, lambda: ({"value": 1}, 3))
        counts = attempt_accounting(self.root)
        self.assertEqual(counts["known_completed_episode_calls"], 3)
        self.assertEqual(counts["interrupted_attempts"], 1)
        self.assertEqual(counts["unknown_interrupted_calls_upper_bound"], 3)

    def test_completed_orphan_work_is_counted_before_cache_write(self):
        path = self.root / "cache.json.gz"
        with patch("evosim_q1005_v01.finalize.write_once", side_effect=RuntimeError("injected artifact write")):
            with self.assertRaises(RuntimeError):
                cached_evaluation(self.root, path, {}, 3, lambda: ({"value": 1}, 3))
        cached_evaluation(self.root, path, {}, 3, lambda: ({"value": 1}, 3))
        self.assertEqual(attempt_accounting(self.root)["known_completed_episode_calls"], 6)

    def test_probe_single_step_accounting_has_separate_unknown_bound(self):
        path = self.root / "probe.json.gz"
        with self.assertRaises(RuntimeError):
            cached_evaluation(self.root, path, {}, 0, lambda: (_ for _ in ()).throw(RuntimeError("probe failed")), single_step_actions=210)
        cached_evaluation(self.root, path, {}, 0, lambda: ({"value": 1}, 0), single_step_actions=210)
        counts = attempt_accounting(self.root)
        self.assertEqual(counts["known_completed_episode_calls"], 0)
        self.assertEqual(counts["known_single_step_actions"], 210)
        self.assertEqual(counts["unknown_single_step_actions_upper_bound"], 210)


if __name__ == "__main__":
    unittest.main()
