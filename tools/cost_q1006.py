"""훈련 자료만 쓰는 별도 비용 표본. 조건·예산 선택에는 사용하지 않는다."""
from pathlib import Path
import time,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'studies/interaction'))
from evosim_q1006_v01.protocol import RESULTS,load
from evosim_q1006_v01.network import Genome,NETWORK
from evosim_q1006_v01.model import Treatment
from evosim_q1006_v01.evaluation import evaluate
from evosim_q1006_v01.search import directory
from evosim_interaction_v02.storage import read_json,write_once
from evosim_interaction_v02.datasets import make_split

def main():
    root=RESULTS/'pilot';config=load(root);out=root/'cost_projection.json'
    if out.exists():raise FileExistsError('Preserve prior cost measurement')
    rows=[]
    for condition in ('F0','F1'):
        for seed in config['seeds']:
            initial=read_json(directory(root,condition,'evolution',seed)/'result.json')['initial_population'][0]
            g=Genome.from_dict(initial,NETWORK)
            for n,repeats in ((24,1),(252,3)):
                start=time.perf_counter()
                metrics,records=evaluate(g,make_split().train[:n],seed,'train:cost',Treatment(condition),repeats=repeats,training=True)
                elapsed=time.perf_counter()-start
                write_once(root/'cost_samples'/f'{condition}_{seed}_{n}.json.gz',{'records':records,'metrics':metrics,'seconds':elapsed})
                rows.append({'condition':condition,'seed':seed,'episodes':len(records),'seconds':elapsed})
    small=sum(r['seconds'] for r in rows if r['episodes']==24)/4
    large=sum(r['seconds'] for r in rows if r['episodes']==756)/4
    kernel=120*20*(100*small+11*large)/2
    result={'source_digest':config['source_digest'],'plan_digest':config['plan_digest'],'rows':rows,
      'additional_training_episode_calls':sum(r['episodes'] for r in rows),'validation_test_calls':0,
      'kernel_seconds_two_workers':kernel,'conservative_kernel_seconds_factor_2':kernel*2,
      'limitations':'initial four-module kernels; structural growth, compression, audits, selection and publication add cost; estimate not deadline',
      'scientific_conditions_changed':False}
    write_once(out,result);print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
