"""본/확인 완료 뒤 기존 로그의 다양성·모듈 수·순위 충돌을 기술한다. 새 정책 호출0."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'studies/interaction'))
from evosim_q1006_v02.protocol import RESULTS,load
from evosim_interaction_v02.storage import read_json,write_once,file_digest

def main():
    out=ROOT/'docs/management/q1_006_log_diagnosis.json'
    if out.exists():raise FileExistsError('Preserve prior diagnostic')
    rows=[]
    for phase in ('main','confirmation'):
        if not (RESULTS/phase/'summary.json').exists():continue
        config=load(RESULTS/phase)
        for job in config['jobs']:
            if job['method']!='evolution':continue
            path=RESULTS/phase/job['condition']/f"evolution_{job['seed']}"
            history=[]
            for generation in range(config['rounds']):
                saved=read_json(path/f'round_{generation:03d}.json.gz');candidates=saved['candidates']
                history.append({'round':generation,'unique_genotypes':len({r['digest'] for r in candidates}),
                  'mean_modules':float(np.mean([len(r['genome']['modules']) for r in candidates])),
                  'mean_ff_edges':float(np.mean([sum(len(v) for v in r['genome']['ff'].values()) for r in candidates])),
                  'reward_vs_completion_elites_changed':saved['ranking_diagnostics']['s0_s1_elite_sets_changed'],
                  'reward_completion_conflict_pairs':saved['ranking_diagnostics']['completion_score_conflict_pairs'],
                  'best_training_success':max(r['training']['success_rate'] for r in candidates)})
            rows.append({'phase':phase,**job,'history':history})
    if not rows:raise ValueError('Completed main required')
    descriptions={}
    for phase in sorted({r['phase'] for r in rows}):
        for condition in ('F0','F1'):
            group=[r for r in rows if r['phase']==phase and r['condition']==condition]
            descriptions[phase+'/'+condition]={
              'final_unique_genotype_mean':float(np.mean([r['history'][-1]['unique_genotypes'] for r in group])),
              'final_module_mean':float(np.mean([r['history'][-1]['mean_modules'] for r in group])),
              'final_ff_edge_mean':float(np.mean([r['history'][-1]['mean_ff_edges'] for r in group])),
              'round_fraction_reward_vs_completion_elites_differ':float(np.mean([h['reward_vs_completion_elites_changed'] for r in group for h in r['history']]))}
    result={'scope':'secondary log diagnosis; no new policy/environment evaluation; not a progress rule',
      'policy_episode_calls':0,'independent_review':False,'rows':rows,'descriptions':descriptions,
      'limitations':'genotypic diversity is not behavioral diversity; selection disagreements and structure-size correlations do not establish causal bottlenecks'}
    raw=RESULTS/'log_diagnosis.json.gz'
    write_once(raw,result)
    index={k:v for k,v in result.items() if k!='rows'}
    index.update(raw_data=raw.relative_to(ROOT).as_posix(),raw_sha256=file_digest(raw),evolution_jobs=len(rows))
    out.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(descriptions,ensure_ascii=False))

if __name__=='__main__':main()
