"""F1 시드1015의 기존 기록에서 행동 대비를 읽는다. 새 정책 계산·시험은 없다."""
from pathlib import Path
import gzip,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'studies/interaction'))
from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.environment import LineCollectEnv

def main():
    out=ROOT/'docs/management/q1_006_f1_1015_behavior_review.json'
    if out.exists():raise FileExistsError('Preserve prior analysis')
    path=ROOT/'studies/interaction/results_q1006_v02/main/F1/evolution_1015'
    saved=json.loads(gzip.decompress((path/'test.json.gz').read_bytes()))['roles']['best']
    result=json.loads((path/'result.json').read_text(encoding='utf-8'))
    lookup={e.episode_id:e for e in make_split().test}
    first={};goals={};times={};visited=set();wrong=set();trace=[];replayed=0
    for row in saved['records']:
        if row['repeat']!=0:continue
        episode=lookup[row['episode_id']];env=LineCollectEnv();env.reset(episode);replayed+=1
        first[episode.start,episode.targets[0]]=int(row['actions'][0])
        for step,a in enumerate(row['actions']):
            st=env.state;p,g,n,b=st.position,st.goal,3-st.collected,24-step;a=int(a)
            if step==0 and a!=(0 if g<p else 1):wrong.add(row['episode_id'])
            visited.add((p,g,n,b));goals.setdefault((p,n,b),{}).setdefault(g,a);times.setdefault((p,g,n),{}).setdefault(b,a)
            transition=env.step(a)
            if row['episode_id']=='0:2,6,4':trace.append({'step':step+1,'before_position':p,'goal':g,'action':a,
              'after_position':env.state.position,'collected':transition.collected,'wall':transition.wall_collision,'failed_collect':transition.failed_collect})
    assert replayed==504 and len(wrong)==187 and len(trace)==18
    goal_cases=[{'position_remaining_goals_budget':k,'actions_by_goal':v} for k,v in goals.items() if len(set(v.values()))>1]
    time_cases=[{'position_goal_remaining_goals':k,'actions_by_remaining_budget':v} for k,v in times.items() if len(set(v.values()))>1]
    payload={'scope':'posthoc interpretation of existing selected candidate; no independent confirmation',
      'seed':1015,'genome_digest':result['best']['digest'],'modules':len(result['best']['genome']['modules']),
      'test_metrics':saved['metrics'],'distinct_test_episodes':504,'repeats':3,'ties':saved['metrics']['ties'],
      'first_noncoordinate_action_distinct_episodes':len(wrong),'unique_visited_states':len(visited),
      'goal_action_varying_groups':len(goal_cases),'time_action_varying_groups':len(time_cases),
      'goal_examples':goal_cases[:3],'time_examples':time_cases[:3],'example_episode_id':'0:2,6,4','example_trace':trace,
      'first_action_table':[{'position':p,'actions_by_goal':[first.get((p,g)) for g in range(7)]} for p in range(7)],
      'best_validation_history':[{'round_zero_based':h['round'],'success_rate':h['best_validation']['success_rate']} for h in result['history'] if h['validation_performed']],
      'episode_logs_replayed_without_policy':504,'new_policy_episode_calls':0,'new_policy_forward_calls':0,'independent_review':False}
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print({k:payload[k] for k in ('modules','distinct_test_episodes','first_noncoordinate_action_distinct_episodes','goal_action_varying_groups','time_action_varying_groups','new_policy_episode_calls')})

if __name__=='__main__':main()
