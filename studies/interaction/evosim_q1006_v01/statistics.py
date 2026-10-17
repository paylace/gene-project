"""40 짝 시드의 평균·발견 비율·조건부 부호 검정을 구분한다."""
import math
from pathlib import Path
import numpy as np
from evosim_interaction_v02.storage import read_json
from .protocol import load
from .search import directory,attempt_accounting


def sign_p(positive,nonzero):
    if type(positive) is not int or type(nonzero) is not int or not 0<=positive<=nonzero:
        raise ValueError('Invalid sign counts')
    return sum(math.comb(nonzero,k) for k in range(positive,nonzero+1))/2**nonzero


def paired_summary(values,draws):
    values=np.asarray(values,dtype=float)
    if values.shape!=(40,) or not np.isfinite(values).all():
        raise ValueError('Forty finite pairs required')
    values=np.where(abs(values)<=1e-12,0,values)
    positive=int((values>0).sum());negative=int((values<0).sum())
    return {'differences':values.tolist(),'mean':float(values.mean()),'median':float(np.median(values)),
      'sd':float(values.std(ddof=1)),'ci95':np.percentile(values[draws].mean(axis=1),[2.5,97.5]).tolist(),
      'positive_seeds':positive,'negative_seeds':negative,'zero_seeds':40-positive-negative,
      'sign_p':sign_p(positive,positive+negative)}


def primary_decision(rates):
    if set(rates)!= {'F0','F1'}:
        raise ValueError('Two complete conditions required')
    arrays={k:np.asarray(v,dtype=float) for k,v in rates.items()}
    if any(v.shape!=(40,) or not np.isfinite(v).all() or (v<0).any() or (v>1).any() for v in arrays.values()):
        raise ValueError('Forty finite rates in [0,1] per condition required')
    draws=np.random.default_rng(72006).integers(0,40,size=(10000,40))
    row=paired_summary(arrays['F0']-arrays['F1'],draws)
    row['progress_gate']=row['mean']>=.1-1e-12 and row['sign_p']<=.05
    discovery=paired_summary((arrays['F0']>=.8).astype(float)-(arrays['F1']>=.8).astype(float),draws)
    return {'primary':{'F0-F1':row},'confirmation_condition':'F0' if row['progress_gate'] else None,
      'discovery_fraction_difference_descriptive':discovery,
      'test_scope':'P(F0>F1 | nonzero paired difference), not a test of the mean',
      'ci_scope':'individual descriptive percentile intervals; no simultaneous coverage',
      'automatic_adoption':False}


def summarize(root):
    root=Path(root);config=load(root)
    if config['phase'] not in ('main','confirmation') or len(config['seeds'])!=40:
        raise ValueError('Completed forty-seed official cohort required')
    rows=[];rates={};descriptions={};work={'search_episode_calls':0,'test_episode_calls':0,'probe_single_state_calls':0,
      'probe_audit_single_state_calls':0,'unknown_interrupted_attempts':0,'unknown_episode_calls_upper_bound':0,
      'unknown_single_step_actions_upper_bound':0}
    mutations={}
    for job in config['jobs']:
        path=directory(root,**job)
        tested=read_json(path/'test.json.gz');probe=read_json(path/'probe.json.gz')
        result=read_json(path/'result.json')
        rate=tested['roles']['best']['metrics']['success_rate']
        rates.setdefault(job['condition'],{}).setdefault(job['method'],[]).append(rate)
        rows.append({**job,'roles':{r:tested['roles'][r]['metrics'] for r in ('best','initial','final')},
          'probe':probe['metrics'],'trajectory':read_json(path/'diagnostics.json'),
          'first_validation_80_percent_round':next((r['round'] for r in result['history'] if r['validation_performed'] and r['best_validation']['success_rate']>=.8),None)})
        for sub,key in ((path,'search_episode_calls'),(path/'test_work','test_episode_calls'),
                        (path/'probe_work','probe_single_state_calls'),(path/'audit_probe_work','probe_audit_single_state_calls')):
            a=attempt_accounting(sub)
            work[key]+=a['known_single_step_actions'] if 'state_calls' in key else a['known_completed_episode_calls']
            work['unknown_interrupted_attempts']+=a['interrupted_attempts']
            work['unknown_episode_calls_upper_bound']+=a['unknown_interrupted_calls_upper_bound']
            work['unknown_single_step_actions_upper_bound']+=a['unknown_single_step_actions_upper_bound']
        counts={'mutation_proposals':0,'changed':0,'weight_changed':0,'structural_changed':0,'forbidden':0,'kinds':{}}
        for g in range(1,config['rounds']):
            saved=read_json(path/f'round_{g:03d}.json.gz')
            for candidate in saved['candidates']:
                mutation=candidate['ancestry']['mutation']
                if mutation:
                    counts['mutation_proposals']+=1
                    counts['changed']+=mutation['changed']
                    counts['weight_changed']+=mutation['changed'] and mutation['kind']=='weight'
                    counts['structural_changed']+=mutation['changed'] and mutation['kind']!='weight'
                    counts['forbidden']+=mutation['proposal_rejected']
                    counts['kinds'][mutation['kind']]=counts['kinds'].get(mutation['kind'],0)+1
        mutations[f"{job['condition']}/{job['method']}_{job['seed']}"]=counts
    for condition,methods in rates.items():
        for method,values in methods.items():
            descriptions[f'{condition}/{method}']={'mean':float(np.mean(values)),'median':float(np.median(values)),
              'sd':float(np.std(values,ddof=1)),'minimum':min(values),'maximum':max(values),
              'at_least_80_percent_seeds':sum(v>=.8 for v in values),'rates':values}
    decision=primary_decision({c:rates[c]['evolution'] for c in config['conditions']})
    draws=np.random.default_rng(72006).integers(0,40,size=(10000,40))
    secondary={c:paired_summary(np.array(rates[c]['evolution'])-rates['F0']['random'],draws) for c in config['conditions']}
    for name in ('blind_work','baseline_work'):
        a=attempt_accounting(root/name)
        work[name+'_episode_calls']=a['known_completed_episode_calls']
        work['unknown_interrupted_attempts']+=a['interrupted_attempts']
        work['unknown_episode_calls_upper_bound']+=a['unknown_interrupted_calls_upper_bound']
        work['unknown_single_step_actions_upper_bound']+=a['unknown_single_step_actions_upper_bound']
    return {'phase':config['phase'],'protocol_digest':config['protocol_digest'],'source_digest':config['source_digest'],
      'rows':rows,'rates':rates,'descriptions':descriptions,'decision':decision,'mutations':mutations,
      'actual_work':work,'secondary_evolution_minus_shared_random_descriptive':secondary,
      'baselines':read_json(root/'baseline_results.json'),'random_control_shared':True,
      'cold_probe_audit_states':441*len(config['jobs']),'benchmark_previously_observed':True}
