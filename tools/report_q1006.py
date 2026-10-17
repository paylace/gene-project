"""완료 집계만 읽어 Q1-006 보고서·시드 CSV를 작성한다. 정책 호출은 없다."""
from pathlib import Path
import csv,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'studies/interaction'
sys.path.insert(0,str(STUDY))
from evosim_interaction_v02.storage import read_json
from evosim_q1006_v02.protocol import load,RESULTS

def main():
    report=ROOT/'docs/reports/Q1-006_고정구조와구조탐색_비교.md'
    csv_path=ROOT/'docs/management/q1_006_seed_metrics.csv'
    if report.exists() or csv_path.exists():raise FileExistsError('Preserve completed report; review edits separately')
    phases=[]
    for phase in ('main','confirmation'):
        if (RESULTS/phase/'summary.json').exists():
            config=load(RESULTS/phase);summary=read_json(RESULTS/phase/'summary.json')
            if len(summary['rows'])!=120 or not read_json(RESULTS/phase/'search_audit.json')['passed'] or not read_json(RESULTS/phase/'test_audit.json')['passed']:
                raise ValueError('Incomplete audited cohort')
            phases.append((config,summary))
    if not phases or phases[0][0]['phase']!='main':raise ValueError('Completed main required')
    main_summary=phases[0][1]
    expected=main_summary['decision']['confirmation_condition'] is not None
    if expected!=(len(phases)==2):raise ValueError('Confirmation status does not match registered gate')
    csv_rows=[]
    text=['# Q1-006 — 고정 구조와 구조 탐색의 비교','',
      '| 항목 | 내용 |','|---|---|',
      '| 버전·분류 | 1.0 / 새 탐색 시드의 사전 조건 고정 비교 |',
      '| 상태 | 실행·작성 완료, 독립 검토 미완료 |',
      '| 수행일·작성일 | 2026-10-17 한국 시간 |',
      '| 질문 | 같은 출발 템플릿에서 구조 변경 허용이 탐색 성과에 미치는 영향 |',
      '| 계획 | [v0.2 계획](../protocols/Q1-006_v02_고정구조와구조탐색_실험계획.md), [수치 명세](../protocols/q1_006_v02_plan.json) |',
      '| 준비 | [구현·예비 검증](../management/Q1-006_v02_실행준비검증.md), [사전 모의 검출력](../management/q1_006_power_simulation.json) |',
      '| 자료 | [본 집계](../../studies/interaction/results_q1006_v02/main/summary.json), [시드 CSV](../management/q1_006_seed_metrics.csv) |','',
      '## 1. 목적·대상·방법','',
      '해결 가능한 공통4모듈 tanh 순방향 템플릿에서 F0는 가중치·편향만, F1은 모듈4~9개와 순방향 연결 변경도 허용했다. 두 조건은 동일한 시드별 초기20개체로 출발했다. 같은 분포의 독립 무작위 탐색을 공통 대조로 실행했다. 수작업 정답 가중치를 심지 않았다. 양쪽 모두 순환·모듈 기억·생애 학습은 없으며 연산은 tanh로 제한했다.',
      '',
      '환경은 기존7칸·순차3목표·24행동, 원래0~1 위치/목표·남은 목표/행동 비율의 네 관측이다. R0 수집+1·매 행동−0.01, S0 훈련 평균 보상으로 번식했다. 집단20·100회차·엘리트20%·토너먼트3·교차0.65·변이0.5를 유지했다. 0.42/0.12/0.06/0.18/0.22/0.10의 변이 종류 가중치를 정규화했다. F0의 구조 제안과 양쪽 add_rec는 무효로 남겼다. 연산·decay는 변경하지 않았고 가중치±3·편향±2를 강제했다.',
      '',
      '비교는 같은 제안 방식·평가 예산의 탐색 방법 전체다. 유효 변경 수·실제 연산량까지 같지 않다. F1에서 큰 구조의 weight 제안은 더 많은 가중치를 바꿀 수 있고 구조별 난수 소비도 갈라진다. 이 차이들은 원자료에 남겼으며 구조의 순수 효과로 과장하지 않는다.',
      '',
      '## 2. 평가·독립 단위·판정','',
      '본40짝 시드1001~1040에 F0진화·F1진화·공통 무작위 탐색을 적용했다. 훈련756·검증252·테스트504에피소드는 기존 benchmark를 재사용했다. 인접 목표가 서로 다른252개 목표 목록과 첫 목표와 다른6개 시작 위치의 총1,512조합이며, 목표 목록 단위와 좌우 반전 쌍을 보존해 분할했다. 매 회차 훈련24에피소드,0·9·19·…·99회차 전 개체 검증252×3반복을 사용했다. 검증 완료율·원래 보상·추가 행동 순으로 best를 선발하고 동률이면 먼저 본 후보를 유지했다. best·0회차 best·마지막 best 및 관측 없는 행동 순서 기준선을 모두 잠근 뒤 테스트504×3반복을 열었다. 동일 역할 유전체는 결과 캐시를 공유했다.',
      '',
      '주 지표는 검증 best의 테스트 완료율, 독립 단위는 탐색 시드40개, 주 대비는 F0−F1이다. 평균 차이≥10%포인트와 동률 제외 한쪽 정확 부호 검정 p≤0.05를 함께 만족할 때만 새40쌍1101~1140에서 같은 비교를 확인하도록 정했다. 이 검정의 대상은 비동률 시드에서 F0가 우세할 확률이며 평균 차이 검정이 아니다. bootstrap10,000회(난수72006)는 짝을 보존한 개별95% 기술 구간이다. 부수 비교·발견 비율로 진행 실패를 뒤집지 않았다.',
      '',
      '완료율80% 이상 후보를 얻은 시드 수/40을 발견 비율로 기록했다. 최초 검증 best80% 도달 회차·누적 탐색 호출은 테스트 성공 판정과 다른 보조 지표다. 테스트 역할·초기 반응성으로 후보를 재선발하지 않았다. 사전 모의 분포에서40쌍도 작은 효과의 검출력이 낮을 수 있음을 기록했으며, 결과를 보고 시드·예산을 늘리지 않았다.','',
      '초기 반응성은 상태를0으로 두고 위치7×목표7×남은 목표3×남은 행동{1,12,24}의441상태에서 측정했다. 반대 이동을 요구하는315쌍의 양쪽에서 좌표 규칙과 맞는 행동이 유일한 최대 확률인지, 수집63상태도 같은 방식으로 검사했다. 동률 난수의 변화는 반응으로 세지 않았다. 이 정답은 좌표 규칙과의 일치 기준이며 남은 행동1 같은 모든 가상 상태에서 보상 최적 행동이 유일하다는 뜻은 아니다.',
      '',
      '## 3. 결과','']
    for config,summary in phases:
        phase=config['phase'];label='본 비교' if phase=='main' else '새 시드 확인'
        text += [f'### {label}','',
          '| 조건·방법 | 평균 완료율% | 중앙값% | 시드 표준편차 pp | 80% 이상 발견 /40 |',
          '|---|---:|---:|---:|---:|']
        for key in ('F0/evolution','F1/evolution','F0/random'):
            d=summary['descriptions'][key]
            text.append(f"| {key} | {100*d['mean']:.4f} | {100*d['median']:.4f} | {100*d['sd']:.4f} | {d['at_least_80_percent_seeds']} |")
        d=summary['decision']['primary']['F0-F1']
        text += ['',f"주 대비 F0−F1 평균 {100*d['mean']:+.4f}pp, 중앙 {100*d['median']:+.4f}pp, 개별95% 기술 구간 [{100*d['ci95'][0]:+.4f}, {100*d['ci95'][1]:+.4f}]pp. 양수/음수/동률 {d['positive_seeds']}/{d['negative_seeds']}/{d['zero_seeds']}, 동률 제외 부호 p={d['sign_p']:.8g}. 사전 진행 기준 {'충족' if d['progress_gate'] else '미충족'}.",'']
        discovery=summary['decision']['discovery_fraction_difference_descriptive']
        text += [f"80% 이상 발견 비율 차이 F0−F1은 {100*discovery['mean']:+.2f}pp, 개별 기술 구간 [{100*discovery['ci95'][0]:+.2f}, {100*discovery['ci95'][1]:+.2f}]pp다. 주 진행 판정과 구분한다.",'',
          '| 조건·방법 | 초기 best 평균 완료율% | 마지막 best 평균 완료율% | 엄격 좌우 쌍 반응 평균% | 수집 유일 정답 평균% |',
          '|---|---:|---:|---:|---:|']
        for condition,method in (('F0','evolution'),('F1','evolution'),('F0','random')):
            subset=[r for r in summary['rows'] if r['condition']==condition and r['method']==method]
            initial=np.mean([r['roles']['initial']['success_rate'] for r in subset]);final=np.mean([r['roles']['final']['success_rate'] for r in subset])
            direction=np.mean([r['probe']['strict_direction_pair_rate'] for r in subset]);collect=np.mean([r['probe']['strict_collection_rate'] for r in subset])
            text.append(f'| {condition}/{method} | {100*initial:.4f} | {100*final:.4f} | {100*direction:.4f} | {100*collect:.4f} |')
        text += ['', '| 조건 | 변이 제안 | 실제 변경 | 가중치/편향 변경 제안 | 구조 변경 제안 | 금지 제안 |', '|---|---:|---:|---:|---:|---:|']
        for condition in ('F0','F1'):
            subset=[v for k,v in summary['mutations'].items() if k.startswith(condition+'/evolution_')]
            totals={key:sum(r[key] for r in subset) for key in ('mutation_proposals','changed','weight_changed','structural_changed','forbidden')}
            text.append('| '+condition+' | '+' | '.join(str(totals[key]) for key in totals)+' |')
        text += ['', '실제 변경은 제안 하나가 유전체 지문을 바꾼 횟수다. 바뀐 개별 가중치 수가 아니다. 교차의 파라미터·간선 변화는 이 변이 제안 집계와 별도이며 원자료의 부모·자손 지문으로 보존했다.','']
        for row in summary['rows']:
            generation=row['first_validation_80_percent_round']
            cost=None if generation is None else 20*((generation+1)*24+sum(v<=generation for v in config['validation_rounds'])*252*3)
            csv_rows.append({'phase':phase,'condition':row['condition'],'method':row['method'],'seed':row['seed'],
              **{r+'_success_rate':row['roles'][r]['success_rate'] for r in ('best','initial','final')},
              'at_least_80_percent':row['roles']['best']['success_rate']>=.8,
              'strict_direction_pair_rate':row['probe']['strict_direction_pair_rate'],
              'strict_collection_rate':row['probe']['strict_collection_rate'],
              'first_validation_80_round':generation,'episode_cost_at_first_validation_80':cost})
        # 초기 유전체 집단이 조건/방법에 따라 달라지지 않았는지 별도 저장 자료 대조.
        for seed in config['seeds']:
            populations=[read_json(RESULTS/phase/c/f'{m}_{seed}'/'result.json')['initial_population'] for c,m in (('F0','evolution'),('F1','evolution'),('F0','random'))]
            if any(p!=populations[0] for p in populations[1:]):raise ValueError('Initial populations differ')
    text += ['### 사후 사례 — 과정 완수와 초기 반응의 차이','',
      'F1 시드1015의100회차 후보는8모듈이며 검증·테스트 완료율이100%였다. 테스트504에피소드×3반복의1,512기록 중 첫 좌표 규칙 행동과 다른561회도 모두 완료했다. 이 사례는 결과를 본 뒤 고른 단일 사례이며 독립 확인이 아니다. 별도 정책 재평가는 하지 않았다.',
      '',
      '반면 이 후보의 균등한 초기 가상 상태에서 엄격 좌우 반응은26/315쌍(8.25%), 수집은10/63상태(15.87%)였다. 가상 상태 분포에는 실제 시작·방문 과정과 다른 남은 목표/행동 조합도 포함된다. 좌표 규칙의 즉시 실행과24행동 안의 완수는 다른 능력이다. 초기 반응 하나로 과정 전체의 실패를 판정할 수 없다. 내부 기억·생애 학습이 없는 정책이므로 회복을 기억이나 경험 학습의 증거로 해석하지 않는다. [사례 기록](../management/q1_006_perfect_candidate_case.json).','']
    diagnosis_path=ROOT/'docs/management/q1_006_log_diagnosis.json'
    if diagnosis_path.exists():
        diagnosis=read_json(diagnosis_path)
        text += ['### 기존 로그의 보조 기술 진단','',
          '새 정책 호출 없이 저장한 회차를 읽었다. 사후 로그 기술이며 진행 기준이나 원인 확정에 사용하지 않는다.','',
          '| 단계·조건 | 마지막 회차 고유 유전체 수 평균 /20 | 마지막 모듈 수 평균 | 마지막 순방향 연결 수 평균 | 보상/완료율 엘리트 집합이 다른 회차 비율% |',
          '|---|---:|---:|---:|---:|']
        for key,d in diagnosis['descriptions'].items():
            text.append(f"| {key} | {d['final_unique_genotype_mean']:.3f} | {d['final_module_mean']:.3f} | {d['final_ff_edge_mean']:.3f} | {100*d['round_fraction_reward_vs_completion_elites_differ']:.3f} |")
        text += ['', '유전체 지문의 다양성은 행동 다양성이 아니다. 순위 불일치와 모듈 수는 병목을 검토할 단서이며 인과 효과를 뜻하지 않는다. [진단 원자료](../management/q1_006_log_diagnosis.json)에 회차별 값을 보존했다.','']
    text += ['## 4. 감사·작업량·오류','',
      '후보 잠금 전에 모든 회차의 초기 집단·유전체·교차/변이·훈련 점수·번식 순위·검증 선발·재개 RNG·ID와 파일 지문을 재구성했다. 행동 기록은 환경·보상·종료로 재생했다. 전체 신경망 출력을 모두 독립 재계산한 감사는 아니다. scalar/batch는 기술 fixture에서 대조했다. 초기441상태의 실제 softmax 동률 집합·출력은 best별 재계산 캐시와 원장으로 별도 검사했다.','']
    for config,summary in phases:
        text += [f"{config['phase']} 실제 작업 원장:",'','```json',json.dumps(summary['actual_work'],ensure_ascii=False,indent=2),'```','']
    text += [
      'v0.1 예비 작업 구성 오류로6작업14,400호출만 생성했다. all_jobs_complete=false로 본 진행 자격이 없었고 본은 미실행이었다. CLI 정상 종료도 보완 대상으로 기록했다. 원본 소스·수치 명세·게이트·원자료와 추가 훈련 비용 표본3,120호출을 보존했다. 새 v0.2에서 예비 양쪽 무작위 대조·미완료 종료를 고쳤다. 신규20+기존173=193검사 뒤8작업19,200훈련 호출의 예비·감사를 통과했다. v0.2 추가 훈련 비용 표본3,120호출도 별도다. 과학 조건·본 시드·판정 변경이나 실패 시드 제외는 없었다.',
      '',
      '단위 검사 안의 작은 합성 탐색·알려진 정답 정책의 저장 fixture는 기술 검사이며 본 성능 표본에 합치지 않는다. 모든 단위 검사의 내부 정책 호출 총량은 별도 일괄 집계하지 않았다. 본·예비·비용 표본·단일 상태 호출은 위와 원자료 원장에서 구분한다.',
      '',
      '사전 등록 커밋의 [GitHub Linux 검사](https://github.com/paylace/gene-project/actions/runs/39005909560)도 통과했다. [코드 대조](../management/q1_006_v02_revision_comparison.json)는 버전 참조를 정규화한 뒤 v0.1→v0.2의 변경 파일이 예비 구성·종료 처리·추가 검사뿐이며 과학적 수치 필드는 같음을 확인했다. 자동 검사는 독립 검토가 아니다.',
      '',
      '## 5. 해석·결론과 다음 질문','']
    if expected:
        passed=phases[-1][1]['decision']['primary']['F0-F1']['progress_gate']
        text.append('본 진행 기준을 충족해 사전 예약한 같은 비교의 새40쌍 확인을 수행했다. 확인 기준은 '+('충족했다. 이 조건·예산에서 F0 우세의 재현 근거를 얻었다.' if passed else '충족하지 못했다. 본의 차이를 안정적인 F0 우세로 확정하지 않는다.'))
    else:
        text.append('본 진행 기준을 충족하지 않아 확인을 실행하지 않았다. 반대 조건으로 확인 대상을 바꾸거나 시드·예산을 추가하지 않았다. 작은 효과·다른 예산·다른 탐색 알고리즘의 가능성을 배제하지 않는다.')
    text += ['',
      '수작업 해법 존재·이번 탐색의 발견 빈도·세대 간 개선을 구분한다. 생애 학습·기억·새 환경 일반화나 일반 지능을 검증한 실험은 아니다. 기존 A0·Q1-009와는 초기 템플릿과 연산 조건이 달라 성적 차이를 단일 원인의 효과라고 부르지 않는다. 평균 하나와 최고 성공 하나로 안정성을 판단하지 않는다.',
      '',
      '다음 단계는 이 결과의 발견 비율·세대별 성능 변화·실제 변이/무효 기록을 함께 검토해, 규모 비교로 진행할 근거가 있는지 판단하는 것이다. 둘 다 낮다면 적합도 신호·변이 크기·다양성 검토를 우선한다. 이번 실패를 이유로 조건을 무제한 확장하지 않는다.',
      '',
      'CSV의 최초80% 회차는0부터 시작하는 회차 번호이며 빈칸은 미도달이다. 해당 비용은 그 검증 회차 전체가 끝날 때까지의 훈련·검증 에피소드 합이다. 테스트80% 성공과 검증 중80% 발견은 서로 다르다.',
      '',
      '## 6. 재현·보존','',
      '동결 v0.2 소스35파일 지문 `d8d2d533a4f8ad4d1eec1c8a394b5150d29e64e1cc58dc1260533c6b40511320`. Python3.12.14·NumPy2.3.5·Windows. 사전 등록 커밋 ec79983에 계획·소스·프로토콜·예비·실패 기록을 게시한 뒤 본을 실행했다. 본 코드·계획·게이트를 소급 변경하지 않았다.',
      '',
      '실행 작업 디렉터리 `studies/interaction/`:', '', '```powershell',
      'python -B -u -X utf8 -m evosim_q1006_v02.run main --gate ../../docs/management/q1_006_v02_execution_gate.json --pilot-report results_q1006_v02/pilot/cost_report.json --workers 2','```','',
      '완료 결과를 읽기 위해 재실행하지 않는다. 보고·CSV 작성 도구는 원자료만 읽고 정책을 평가하지 않는다. 큰 원자료는 별도 Release와 해시 목록·안전 복원 도구로 보존하며 두 예비 버전과 실패 기록도 포함한다.',
      '',
      '## 수행·작성·검토 기록','',
      '- 연구 방향·실험 요청자: 사용자. 계획대로 실험을 계속하도록 요청함.',
      '- 실험 수행자: Codex 에이전트. 계획·새 버전 구현·검사·예비·본·조건부 확인 판정·감사·집계·보관을 로컬 도구로 수행함.',
      '- 실험 수행일: 2026-10-17 한국 시간.',
      '- 보고서 작성자: Codex 에이전트.',
      '- 문서 작성일 또는 정리일: 2026-10-17 한국 시간.',
      '- 모델·버전: 실행 당시 정확한 호출 모델 ID 미기록.',
      '- 독립 검토 상태: 독립 검토 기록 없음. 자동 검사·원자료 감사와 구분함.','']
    with csv_path.open('x',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    report.write_text('\n'.join(text),encoding='utf-8',newline='\n')
    print({'phases':len(phases),'rows':len(csv_rows),'report':report.name,'policy_calls':0})

if __name__=='__main__':main()
