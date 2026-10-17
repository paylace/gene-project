import copy
import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.policy import NetworkPolicy,BatchNetwork
from evosim_interaction_v02.storage import read_json,write_json
from evosim_q1006_v01.network import (Genome,Module,ModuleIds,NETWORK,initial_genome,mutate,
    validate_condition,topology,crossover)
from evosim_q1006_v01.model import Treatment,breed
from evosim_q1006_v01.protocol import prepare,check_access,load,validate_plan
from evosim_q1006_v01.source import ROOT,plan
from evosim_q1006_v01.search import run_search,directory,attempt_accounting
from evosim_q1006_v01.audit import audit_job,verify_trajectories
from evosim_q1006_v01.evaluation import evaluate,replay_record
from evosim_q1006_v01.probe import probe
from evosim_q1006_v01.analysis import sign_p,primary_decision


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        temp=ROOT/'tools/.verification_tmp';temp.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=temp);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_initialization_identical_across_conditions_and_snapshot_lossless(self):
        for seed in range(10):
            a=initial_genome(random.Random(seed));b=initial_genome(random.Random(seed))
            self.assertEqual(a.to_dict(),b.to_dict())
            validate_condition(a,'F0');validate_condition(b,'F1')
            self.assertEqual(Genome.restore(a.export(),NETWORK).to_dict(),a.to_dict())
        changed=a.export();changed['settings']['condition']='A0'
        with self.assertRaises(ValueError):Genome.restore(changed,NETWORK)

    def test_genome_rejects_other_ops_rec_edges_and_out_of_range(self):
        base=initial_genome(random.Random(0)).to_dict()
        for change in ('op','rec','weight','bias'):
            d=copy.deepcopy(base)
            if change=='op':d['modules']['m0']['op']='relu'
            if change=='rec':d['rec']={'m0':{'m0':1}}
            if change=='weight':d['ff']['m0']['in_0']=3.1
            if change=='bias':d['modules']['m0']['bias']=2.1
            with self.assertRaises(ValueError):Genome.from_dict(d,NETWORK)

    def test_f0_structural_proposals_are_counted_noops(self):
        g=initial_genome(random.Random(0));before=g.digest
        for kind in ('add_module','remove_module','add_ff','add_rec','remove_edge'):
            result=mutate(g,random.Random(2),ModuleIds(),'F0',kind)
            self.assertEqual(g.digest,before);self.assertFalse(result['changed'])
            self.assertTrue(result['proposal_rejected']);self.assertEqual(result['kind'],kind)

    def test_f1_mutations_keep_tanh_dag_bounds_and_monotone_ids(self):
        g=initial_genome(random.Random(0));rng=random.Random(13);allocator=ModuleIds();ids=[]
        for i in range(400):
            result=mutate(g,rng,allocator,'F1',('add_module','weight','add_ff','remove_edge','remove_module','add_rec')[i%6])
            validate_condition(g,'F1')
            if result['added_module']:ids.append(int(result['added_module'][1:]))
        self.assertEqual(ids,sorted(set(ids)));self.assertTrue(ids)
        self.assertEqual(g.rec,{})

    def test_crossover_keeps_dominant_module_set_and_f0_topology(self):
        a=initial_genome(random.Random(1));b=initial_genome(random.Random(2))
        mutate(b,random.Random(3),ModuleIds(),'F1','add_module')
        child=crossover(a,b,random.Random(4),0,1,'F1')
        self.assertEqual(set(child.modules),set(b.modules))
        a=initial_genome(random.Random(1));b=initial_genome(random.Random(2))
        for i in range(20):self.assertEqual(topology(crossover(a,b,random.Random(i),1,0,'F0')),topology(a))

    def test_reproduction_preserves_f0_fixed_topology_and_records_real_changes(self):
        rng=random.Random(2);pop=[initial_genome(rng) for _ in range(20)];ids=ModuleIds()
        for _ in range(8):
            pop,ancestry=breed(pop,[(i,) for i in range(20)],rng,ids,'F0')
            for genome,row in zip(pop,ancestry):
                validate_condition(genome,'F0');self.assertEqual(genome.digest,row['child_digest'])
                if row['mutation'] and row['mutation']['changed']:self.assertEqual(row['mutation']['kind'],'weight')

    def test_scalar_batch_outputs_and_full_environment_records_agree(self):
        g=initial_genome(random.Random(5));before=g.digest
        episodes=make_split().train[:24]
        scalar=NetworkPolicy(g,NETWORK);obs=np.array([[e.start/6,e.targets[0]/6,1,1] for e in episodes])
        expected=[]
        for o in obs:scalar.reset();expected.append(scalar.forward(o))
        np.testing.assert_allclose(BatchNetwork(g,len(obs),NETWORK).forward(obs),expected,rtol=1e-13,atol=1e-13)
        metrics,records=evaluate(g,episodes,41,'train:0',Treatment('F0'),training=True)
        for e,r in zip(episodes,records):self.assertEqual(replay_record(e,r)[0],r['progress_units'])
        self.assertEqual(verify_trajectories(records,{e.episode_id:e for e in episodes},training=True)['episodes'],24)
        self.assertEqual(g.digest,before)
        other,other_records=evaluate(g,episodes,41,'train:0',Treatment('F1'),training=True)
        self.assertEqual((metrics,records),(other,other_records))

    def test_probe_zero_policy_cannot_pass_unique_direction_or_collection(self):
        g=initial_genome(random.Random(5))
        for edges in g.ff.values():
            for node in edges:edges[node]=0
        metrics,records=probe(g)
        self.assertEqual(len(records),441);self.assertEqual(metrics['strict_direction_pair_rate'],0)
        self.assertEqual(metrics['strict_collection_rate'],0)
        self.assertAlmostEqual(metrics['mean_correct_tie_probability'],1/3)

    def test_registered_budget_and_no_test_access_in_pilot(self):
        validate_plan(plan())
        config={'validation_rounds':[],'phase':'pilot'}
        check_access(config,'train:0',make_split().train[:1])
        for purpose,episodes in [('test',make_split().test),('validation',make_split().validation),('train:0',make_split().test)]:
            with self.assertRaises(ValueError):check_access(config,purpose,episodes)
        changed=plan();changed['budgets']['main_search_episodes']-=1
        with self.assertRaises(ValueError):validate_plan(changed)

    def test_resume_and_audit_both_conditions_both_methods(self):
        cfg=prepare(self.root/'resumed','verification',verification_rounds=3,verification_validation=True)
        fresh=prepare(self.root/'fresh','verification',verification_rounds=3,verification_validation=True)
        for job in cfg['jobs']:
            run_search(self.root/'resumed',**job,pause_after=1)
            resumed=run_search(self.root/'resumed',**job);direct=run_search(self.root/'fresh',**job)
            for key in ('best','initial','final','history','initial_population'):self.assertEqual(resumed[key],direct[key])
            path=directory(self.root/'resumed',**job);before=attempt_accounting(path)
            self.assertTrue(audit_job(self.root/'resumed',**job)['passed'])
            self.assertEqual(before,attempt_accounting(path))
        a=read_json(directory(self.root/'resumed','F0','random',41)/'result.json')
        b=read_json(directory(self.root/'resumed','F1','random',41)/'result.json')
        self.assertEqual(a['best']['digest'],b['best']['digest'])
        self.assertEqual(a['initial_population'],b['initial_population'])

    def test_changed_completed_round_rejected_on_resume(self):
        cfg=prepare(self.root,'verification',verification_rounds=1,verification_conditions=['F0'],verification_methods=['evolution'])
        job=cfg['jobs'][0];run_search(self.root,**job)
        path=directory(self.root,**job)/'round_000.json.gz';d=read_json(path);d['candidates'][0]['training']['success_rate']=1
        write_json(path,d)
        with self.assertRaises(ValueError):run_search(self.root,**job)

    def test_sign_ties_excluded_and_mean_cannot_replace_direction(self):
        self.assertEqual(sign_p(0,0),1);self.assertEqual(sign_p(5,5),1/32)
        good=primary_decision({'F0':[1]*10+[0]*30,'F1':[0]*40})
        self.assertTrue(good['primary']['F0-F1']['progress_gate'])
        reverse=primary_decision({'F0':[0]*40,'F1':[1]*40})
        self.assertIsNone(reverse['confirmation_condition'])
        weak=primary_decision({'F0':[.01]*40,'F1':[0]*40})
        self.assertFalse(weak['primary']['F0-F1']['progress_gate'])

    def test_discovery_threshold_secondary_and_cohort_size_strict(self):
        row=primary_decision({'F0':[.8]*40,'F1':[.799]*40})
        self.assertFalse(row['primary']['F0-F1']['progress_gate'])
        self.assertEqual(row['discovery_fraction_difference_descriptive']['mean'],1)
        for values in ([0]*39,[float('nan')]*40,[1.1]*40):
            with self.assertRaises(ValueError):primary_decision({'F0':values,'F1':[0]*40})

    def test_unknown_interrupted_calls_keep_upper_bound(self):
        write_json(self.root/'attempts/a.json',{'status':'started','maximum_episode_calls':100,'maximum_single_step_actions':441})
        row=attempt_accounting(self.root)
        self.assertEqual(row['unknown_interrupted_calls_upper_bound'],100)
        self.assertEqual(row['unknown_single_step_actions_upper_bound'],441)

    def test_manual_solution_is_representable_and_solves_all_known_combinations(self):
        import gzip,json
        data=json.loads(gzip.decompress((ROOT/'studies/interaction/q1006_preflight_v01/template_check.json.gz').read_bytes()))
        genome=Genome.from_dict(data['genome'],NETWORK);validate_condition(genome,'F0')
        split=make_split();episodes=(*split.train,*split.validation,*split.test)
        metrics,records=evaluate(genome,episodes,41,'technical_template',Treatment('F0'))
        self.assertEqual(metrics['success_rate'],1);self.assertEqual(metrics['success_extra_steps'],0)
        self.assertEqual(probe(genome)[0]['strict_direction_pair_rate'],1)

    def test_main_and_confirmation_aggregation_from_saved_fixture_only(self):
        from evosim_q1006_v01.statistics import summarize
        for phase in ('main','confirmation'):
            root=self.root/phase;jobs=[]
            for condition,method in (('F0','evolution'),('F1','evolution'),('F0','random')):
                for seed in range(40):
                    job={'condition':condition,'method':method,'seed':seed};jobs.append(job)
                    path=directory(root,**job);metrics={'success_rate':.8 if condition=='F0' and method=='evolution' else .1}
                    write_json(path/'test.json.gz',{'roles':{r:{'metrics':metrics} for r in ('best','initial','final')}})
                    write_json(path/'probe.json.gz',{'metrics':{'states':441}})
                    write_json(path/'result.json',{'history':[{'round':0,'validation_performed':True,'best_validation':metrics}]})
                    write_json(path/'diagnostics.json',{'fixture':True})
            write_json(root/'baseline_results.json',{'fixture':True})
            config={'phase':phase,'protocol_digest':'fixture','source_digest':'fixture','seeds':list(range(40)),'jobs':jobs,'rounds':1,'conditions':['F0','F1']}
            with patch('evosim_q1006_v01.statistics.load',return_value=config):result=summarize(root)
            self.assertEqual(len(result['rows']),120);self.assertTrue(result['decision']['primary']['F0-F1']['progress_gate'])
            self.assertEqual(result['descriptions']['F0/evolution']['at_least_80_percent_seeds'],40)

if __name__=='__main__':unittest.main()
