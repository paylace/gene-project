"""실제 정책 평가 없이 사전 부호 판정의 희소 성공 검출력을 모의 계산한다."""
from pathlib import Path
import json,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'docs/management/q1_006_power_simulation.json'
    if out.exists():raise FileExistsError('Preserve completed planning simulation')
    rng=np.random.default_rng(86006)
    cases=[]
    for n in (20,40):
        for a,b in ((.05,.05),(.15,.05),(.30,.10),(.50,.10)):
            # 각 조건이0/100%로 양극화된 독립 잠재 결과를 가정한다. 실제 분포 추정이 아니다.
            delta=(rng.random((20000,n))<a).astype(int)-(rng.random((20000,n))<b)
            positive=(delta>0).sum(axis=1);negative=(delta<0).sum(axis=1)
            p=np.array([sum(math.comb(int(x+y),k) for k in range(int(x),int(x+y)+1))/2**int(x+y) for x,y in zip(positive,negative)])
            gate=(delta.mean(axis=1)>=.1-1e-12)&(p<=.05)
            cases.append({'n':n,'F0_success_probability_assumed':a,'F1_success_probability_assumed':b,
              'replicates':20000,'registered_gate_probability':float(gate.mean()),'nonzero_mean':float((positive+negative).mean())})
    result={'seed':86006,'sample_size_chosen_before_real_search':40,'cases':cases,
      'assumptions':'independent Bernoulli all-or-none policy success within pairs; not fitted to Q1-006 data',
      'interpretation':'conditional sign test and mean threshold target different quantities; weak effects can remain undetected',
      'policy_episode_calls':0,'scientific_independent_review':False}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
