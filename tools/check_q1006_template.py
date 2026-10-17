"""Q1-006 설계 전 수작업 tanh 템플릿의 기술 정합성 검사. 탐색은 없다."""
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'studies/interaction'))
from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.network import Genome,Module
from evosim_interaction_v02.storage import write_once
from evosim_q1009_v01.model import SETTINGS
from evosim_q1009_v01.evaluation import evaluate
from evosim_q1009_v01.probe import probe
from evosim_q1009_v01.audit import audit_records
from evosim_q1009_v01.protocol import sources
from evosim_interaction_v02.storage import manifest_digest


def main():
    out=ROOT/'studies/interaction/q1006_preflight_v01/template_check.json.gz'
    if out.exists():raise FileExistsError('Completed technical fixture is preserved')
    modules={f'm{i}':Module('tanh',.1 if i==2 else 0.0,.5) for i in range(4)}
    ff={f'm{i}':{f'in_{j}':0.0 for j in range(4)} for i in range(4)}
    ff['m0'].update(in_0=2.0,in_1=-2.0)
    ff['m1'].update(in_0=-2.0,in_1=2.0)
    for i in range(3):ff[f'out_{i}']={f'm{j}':float(i==j) for j in range(4)}
    genome=Genome(modules,ff,{},SETTINGS)
    assert all(-3<=w<=3 for edges in ff.values() for w in edges.values())
    episodes=(*make_split().train,*make_split().validation,*make_split().test)
    metrics,records=evaluate(genome,episodes,76606,'q1006:technical_template','C0N0')
    probe_metrics,probe_records=probe(genome,'C0N0')
    audited=audit_records(records,episodes)
    passed=metrics['success_rate']==1 and metrics['success_extra_steps']==0 and probe_metrics['strict_direction_pair_rate']==1 and probe_metrics['strict_collection_rate']==1
    result={'scope':'manual technical fixture before Q1-006; not evolved or selected performance',
        'created_utc':datetime.now(timezone.utc).isoformat(),'passed':passed,'genome':genome.to_dict(),
        'modules':4,'dense_input_module_edges':16,'dense_module_output_edges':12,'recurrence':False,
        'operator':'tanh','coordinate_encoding':'original0..1','metrics':metrics,'records':records,
        'probe':probe_metrics,'probe_records':probe_records,'trajectory_audit':audited,
        'evaluation_episode_calls':1512,'cold_states':441,'new_searches':0,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dependency_source_digest':manifest_digest(sources()),'independent_review':False}
    write_once(out,result)
    print({'passed':passed,'episodes':1512,'cold_states':441,'new_searches':0},flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
