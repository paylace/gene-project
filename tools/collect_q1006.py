"""완료 원자료의 실제 호출·감사·소스·시간 구간을 기록한다. 정책 호출0."""
from datetime import datetime,timezone
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'studies/interaction'))
from evosim_q1006_v02.protocol import RESULTS,load
from evosim_q1006_v02.analysis import primary_decision
from evosim_interaction_v02.storage import read_json,file_digest

def main():
    out=ROOT/'docs/management/q1_006_completion.json'
    if out.exists():raise FileExistsError('Preserve completed accounting')
    main_summary=read_json(RESULTS/'main/summary.json')
    phases=('main','confirmation') if main_summary['decision']['confirmation_condition'] else ('main',)
    rows=[]
    for phase in phases:
        root=RESULTS/phase;config=load(root);summary=read_json(root/'summary.json')
        if summary['decision']!=primary_decision({c:summary['rates'][c]['evolution'] for c in ('F0','F1')}):
            raise ValueError('Registered decision recomputation differs')
        if len(summary['rows'])!=120 or any(not read_json(root/name)['passed'] for name in ('search_audit.json','test_audit.json')):
            raise ValueError('Incomplete audited cohort')
        attempts=list(root.glob('*/*/attempts/000_*.json'))
        start=min(datetime.fromisoformat(read_json(p)['started_utc']) for p in attempts)
        end=datetime.fromtimestamp((root/'summary.json').stat().st_mtime,timezone.utc)
        rows.append({'phase':phase,'jobs':120,'paired_seeds':40,'source_digest':config['source_digest'],
          'protocol_digest':config['protocol_digest'],'first_attempt_utc':start.isoformat(),'summary_written_utc':end.isoformat(),
          'first_attempt_to_summary_seconds':(end-start).total_seconds(),'actual_work':summary['actual_work'],
          'decision':summary['decision'],'candidate_lock_sha256':file_digest(root/'candidate_lock.json'),
          'search_audit_sha256':file_digest(root/'search_audit.json'),'test_audit_sha256':file_digest(root/'test_audit.json'),
          'summary_sha256':file_digest(root/'summary.json')})
    result={'experiment_id':'Q1-006','version':'q1006-v0.2','preregistration_commit':'ec7998344cf034dc7552ae5481b1cf84108ac6a4',
      'phases':rows,'failed_v01_pilot_episode_calls':14400,'v02_pilot_episode_calls':19200,
      'additional_training_cost_sample_calls_both_versions':6240,'posthoc_collection_policy_episode_calls':0,
      'unit_fixture_policy_call_total':'not aggregated; excluded from official performance cohort',
      'independent_review':False}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({p['phase']:{'jobs':p['jobs'],'seconds':p['first_attempt_to_summary_seconds'],'decision':p['decision']['primary']['F0-F1']} for p in rows},ensure_ascii=False))

if __name__=='__main__':main()
