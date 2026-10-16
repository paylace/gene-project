import copy
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.environment import LineCollectEnv,EpisodeSpec
from evosim_interaction_v02.policy import NetworkPolicy
from evosim_interaction_v02.evaluation import evaluate as original_evaluate
from evosim_interaction_v02.rollout import derive_seed
from evosim_interaction_v02.storage import read_json,write_json,file_digest
from evosim_q1009_v01.model import SETTINGS,CONDITIONS,initial,transform,observations,control
from evosim_q1009_v01.probe import probe,choices
from evosim_q1009_v01.evaluation import evaluate,summarize_records
from evosim_q1009_v01.audit import audit_records,audit_job
from evosim_q1009_v01.protocol import ROOT,prepare,load
from evosim_q1009_v01.execute import execute,accounting,begin
from evosim_q1009_v01.analysis import paired,distribution


def scalar(genome,episode,seed,purpose,condition,rep=0,sequence=None):
    env = LineCollectEnv()
    obs = env.reset(episode)
    policy = NetworkPolicy(genome,SETTINGS)
    rng = random.Random(derive_seed(purpose,seed,episode.episode_id,rep))
    log,wall,bad,rewards,peak = [],[],[],[],0.0
    while True:
        if sequence is None:
            # act를 직접 사용하고 해당 forward를 한 번만 관찰한다.
            actual_forward = policy.forward
            def capture(x):
                values = actual_forward(x)
                capture.values = values
                return values
            policy.forward = capture
            action = policy.act(observations(obs,condition).tolist(),rng)
            peak = max(peak,max(abs(v) for v in capture.values))
            policy.forward = actual_forward
        else:
            action = sequence[len(log)]
        result = env.step(action)
        log.append(str(action));wall.append(result.wall_collision);bad.append(result.failed_collect)
        rewards.append(round(100*result.reward))
        if result.terminated:
            break
        obs = result.observation
    same = len(log)>=6 and len(set(log[-6:]))==1
    kind = "wall" if same and all(wall[-6:]) else "failed_collect" if same and all(bad[-6:]) else "other"
    return {"episode_id":episode.episode_id,"repeat":rep,"actions":"".join(log),"rewards_cents":rewards,
            "success":env.state.collected==3,"collected":env.state.collected,"walls":sum(wall),
            "failed_collect":sum(bad),"optimal_steps":episode.optimal_steps,"final_position":env.state.position,
            "tail_identical":same,"tail_kind":kind,"peak_abs_logit":peak}


class ProbeTests(unittest.TestCase):
    def test_coordinate_control_is_perfect_in_all_conditions(self):
        for c in CONDITIONS:
            metrics,records = probe(transform(control("coordinate"),c),c)
            self.assertEqual(metrics["strict_direction_pairs_correct"],315)
            self.assertEqual(metrics["strict_collection_correct"],63)
            self.assertEqual(metrics["mean_correct_tie_probability"],1)
            self.assertEqual(len(records),441)

    def test_input_independent_ties_cannot_pass_strict_direction_or_collection(self):
        m,_ = probe(control("zero"),"C0N0")
        self.assertEqual(m["strict_direction_pair_rate"],0)
        self.assertEqual(m["strict_collection_rate"],0)
        self.assertAlmostEqual(m["mean_correct_tie_probability"],1/3)

    def test_constant_controls_have_expected_denominators(self):
        for name in ("left","right","collect"):
            m,_ = probe(control(name),"C0N0")
            self.assertEqual(m["strict_direction_pair_rate"],0)
            self.assertAlmostEqual(m["mean_correct_tie_probability"],1/7 if name=="collect" else 3/7)
            self.assertEqual(m["strict_collection_rate"],int(name=="collect"))

    def test_centering_changes_only_coordinates_and_never_input_array(self):
        raw = np.array([[.25,.75,1,.5]])
        before = raw.copy()
        self.assertEqual(observations(raw,"C1N0").tolist(),[[-.5,.5,1,.5]])
        np.testing.assert_array_equal(raw,before)

    def test_only_output_self_edges_removed_from_cloned_genome(self):
        base = initial(41);before = base.to_dict();new = transform(base,"C0N1")
        self.assertEqual(base.to_dict(),before)
        self.assertEqual(new.ff,base.ff)
        self.assertEqual(new.modules,base.modules)
        expected = copy.deepcopy(base.rec)
        for out in ("out_0","out_1","out_2"):expected[out].pop(out)
        self.assertEqual(new.rec,expected)

    def test_output_recurrence_has_no_cold_start_effect(self):
        for c in ("0","1"):
            base = initial(41)
            self.assertEqual(probe(transform(base,f"C{c}N0"),f"C{c}N0"),probe(transform(base,f"C{c}N1"),f"C{c}N1"))

    def test_probability_tie_rule_is_offset_invariant_and_matches_actual_act(self):
        self.assertEqual(choices([[900.0000005,900,0]]),[[0]])
        self.assertEqual(choices([[.0000005,0,-900]]),[[0]])
        for row in ([0,0,0],[1,1,0],[900.0000005,900,0]):
            selected = choices([row])[0]
            class Fixed:
                settings=SETTINGS
                def forward(self,x):return row
            for seed in range(10):
                self.assertEqual(NetworkPolicy.act(Fixed(),None,random.Random(seed)),random.Random(seed).choice(selected))

    def test_invalid_condition_is_rejected(self):
        with self.assertRaises(ValueError):observations([0,0,0,0],"C2N0")


class EvaluationTests(unittest.TestCase):
    def test_unmodified_condition_matches_frozen_actions_metrics_and_rng(self):
        for seed in (41,42):
            g = initial(seed);eps = make_split().train[:24]
            old,raw = original_evaluate(g,eps,seed,"fixture",SETTINGS,repeats=2)
            new,records = evaluate(g,eps,seed,"fixture","C0N0",repeats=2)
            for k,v in old.items():self.assertEqual(new[k],v)
            for a,b in zip(raw,records):self.assertEqual(a,{k:b[k] for k in a})

    def test_scalar_actual_act_matches_batch_all_conditions(self):
        for seed in (41,42):
            for c in CONDITIONS:
                g = transform(initial(seed),c);before=g.digest;eps=make_split().train[:10]
                _,records = evaluate(g,eps,seed,"fixture",c)
                for ep,record in zip(eps,records):
                    reference = scalar(g,ep,seed,"fixture",c)
                    for k,v in reference.items():
                        if k=="peak_abs_logit":self.assertTrue(math.isclose(v,record[k],rel_tol=1e-12,abs_tol=1e-12))
                        else:self.assertEqual(v,record[k])
                self.assertEqual(before,g.digest)

    def test_wall_and_failed_collect_controls_are_unproductive(self):
        eps=make_split().train[:24]
        for name in ("left","right","collect"):
            m,r = evaluate(control(name),eps,41,"fixture","C0N0")
            self.assertEqual(m["unproductive_tail_rate"],1)
            audit_records(r,eps)

    def test_same_tail_moving_toward_goal_is_not_mistaken_for_walls(self):
        ep=EpisodeSpec(6,(0,1,0));g=control("zero")
        r=scalar(g,ep,41,"fixture","C0N0",sequence=[2]*18+[0]*6)
        self.assertTrue(r["tail_identical"]);self.assertEqual(r["tail_kind"],"other")
        audit_records([r],[ep])

    def test_coordinate_control_solves_complete_dataset_in_optimal_steps(self):
        eps=(*make_split().train,*make_split().validation,*make_split().test)
        for c in CONDITIONS:
            m,r=evaluate(control("coordinate"),eps,41,"fixture",c)
            self.assertEqual(m["success_rate"],1);self.assertEqual(m["unproductive_tail_rate"],0)
            self.assertTrue(all(len(row["actions"])==ep.optimal_steps for row,ep in zip(r,eps)))
            audit_records(r,eps)

    def test_record_corruption_and_coverage_are_rejected(self):
        eps=make_split().train[:2];_,records=evaluate(initial(41),eps,41,"fixture","C0N0")
        for key,value in (("tail_kind","invented"),("rewards_cents",[0]*24),("peak_abs_logit",float("nan"))):
            altered=copy.deepcopy(records);altered[0][key]=value
            with self.assertRaises((ValueError,AssertionError)):audit_records(altered,eps)
        with self.assertRaises(ValueError):audit_records(records[::-1],eps)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        folder=ROOT/"tools/.verification_tmp";folder.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=folder);self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        prepare(self.root,"pilot",fixture=True,fixture_episodes=2)

    def test_completed_resume_reuses_all_work_and_preserves_bytes(self):
        first=execute(self.root,41);before=accounting(self.root/"work/seed_41")
        with patch("evosim_q1009_v01.execute.evaluate",side_effect=AssertionError("must not evaluate")):
            self.assertEqual(execute(self.root,41),first)
        self.assertEqual(before,accounting(self.root/"work/seed_41"))
        self.assertEqual(before["known_episode_calls"],8)
        self.assertEqual(accounting(self.root/"audit_work/seed_41")["known_probe_states"],1764)

    def test_result_corruption_rejected_before_cache_reuse(self):
        execute(self.root,41);p=self.root/"raw/seed_41.json.gz";r=read_json(p);r["seed"]=42;write_json(p,r)
        with self.assertRaises(ValueError):execute(self.root,41)

    def test_audit_corruption_rejected(self):
        execute(self.root,41);p=self.root/"audits/seed_41.json";r=read_json(p);r["passed"]=False;write_json(p,r)
        with self.assertRaises(ValueError):execute(self.root,41)

    def test_completed_work_before_raw_write_is_counted_on_resume(self):
        import evosim_q1009_v01.execute as module
        original=module.write_once
        def fail(path,value):
            if Path(path).suffix==".gz":raise RuntimeError("injected raw write failure")
            return original(path,value)
        with patch.object(module,"write_once",side_effect=fail):
            with self.assertRaises(RuntimeError):execute(self.root,41)
        self.assertEqual(accounting(self.root/"work/seed_41")["known_episode_calls"],8)
        execute(self.root,41)
        self.assertEqual(accounting(self.root/"work/seed_41")["known_episode_calls"],16)

    def test_unknown_work_has_episode_and_probe_upper_bounds(self):
        begin(self.root/"unknown",8,1764)
        a=accounting(self.root/"unknown")
        self.assertEqual(a["known_episode_calls"],0);self.assertEqual(a["unknown_episode_calls_upper"],8)
        self.assertEqual(a["unknown_probe_states_upper"],1764)

    def test_source_plan_runtime_and_protocol_changes_rejected(self):
        import evosim_q1009_v01.protocol as module
        with patch.object(module,"sources",return_value={}):
            with self.assertRaises(ValueError):load(self.root)
        with patch.object(module,"runtime",return_value={}):
            with self.assertRaises(ValueError):load(self.root)
        p=self.root/"protocol.json";r=read_json(p);r["seeds"]=[999];write_json(p,r)
        with self.assertRaises(ValueError):load(self.root)

    def test_out_of_protocol_job_is_rejected(self):
        with self.assertRaises(ValueError):execute(self.root,999)
        with self.assertRaises(ValueError):execute(self.root,73009,"coordinate")

    def test_raw_metrics_and_probe_are_audited(self):
        execute(self.root,41);r=read_json(self.root/"raw/seed_41.json.gz");eps=make_split().train[:2]
        for scope,key,value in (("metrics","success_rate",.123),("probe","strict_direction_pair_rate",.123)):
            altered=copy.deepcopy(r);altered["conditions"][0][scope][key]=value
            with self.assertRaises(ValueError):audit_job(altered,eps)

    def test_main_cannot_start_without_matching_gate(self):
        from evosim_q1009_v01.protocol import RESULTS
        with self.assertRaises(ValueError):prepare(RESULTS/"main","main")


class AnalysisTests(unittest.TestCase):
    def test_constant_paired_effect_interval_and_zero_ties(self):
        draws=np.random.default_rng(1).integers(0,20,size=(100,20))
        a=paired([.1]*20,draws);self.assertAlmostEqual(a["ci95"][0],.1);self.assertEqual(a["positive"],20)
        b=paired([0]*20,draws);self.assertEqual(b["zero"],20);self.assertEqual(b["ci95"],[0,0])

    def test_undefined_success_only_metric_is_preserved(self):
        result=distribution([None,None,.5]);self.assertEqual(result["mean"],.5)
        self.assertEqual(result["missing"],2);self.assertEqual(result["values"],[None,None,.5])


if __name__=="__main__":unittest.main()
