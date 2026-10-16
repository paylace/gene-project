"""완료된 Q1-005 집계에서 표를 작성한다. 성능에 따른 조건 변경은 없다."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "studies/interaction/results_q1005_v01"
REPORT = ROOT / "docs/reports/Q1-005_보상과번식_분리비교.md"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x):
    return f"{100*x:.4f}"


def main():
    if REPORT.exists():
        raise FileExistsError("Report already exists; review revisions explicitly")
    main = read(DATA / "main/summary.json")
    selected = main["decision"]["confirmation_condition"]
    phases = ["main", "confirmation"] if selected else ["main"]
    summaries = {phase: read(DATA / phase / "summary.json") for phase in phases}
    completions = {phase: read(ROOT / f"docs/management/q1_005_{phase}_completion.json") for phase in phases}
    protocol = read(DATA / "main/protocol.json")
    lines = ["# Q1-005 — 보상과 번식 기준의 분리 비교", "",
        "| 항목 | 내용 |", "|---|---|", "| 버전·분류 | 1.0 / 사전 조건 고정 본 비교 및 조건부 확인 |",
        "| 상태 | 작성 완료·독립 검토 미완료 |", "| 수행일·작성일 | 2026-10-16 한국 시간 |",
        "| 상위 질문 | 현재 낮은 완수율에 적합도 신호와 번식 기준이 미치는 영향 |",
        "| 사전 계획 | [개별 계획](../protocols/Q1-005_보상과번식_실험계획.md), 새 조건 성능 확인 전 고정 |",
        "| 선행 결과 | [Q1-004](Q1-004_구조제약_재진화비교.md)의 구조 개선 확인 실패 |",
        "| 원자료 | [본 집계](../../studies/interaction/results_q1005_v01/main/summary.json), [보관 목록](../../evidence/q1_005_v01.json) |", "",
        "## 1. 질문과 방법", "",
        "네트워크 크기와 구조는 A0로 유지하고, 훈련 중 번식에 사용하는 보상과 선호 순서만2×2로 바꿨다. 생애 학습은 OFF다. 환경은7칸에서 순서가 있는 목표3개를24행동 안에 수집하는 과제이며, 입력은 정규화한 현재 위치·현재 목표 위치·남은 목표 비율·남은 행동 비율4개, 행동은 왼쪽·오른쪽·수집3개다. 이후 목표 정보는 현재 목표를 수집한 뒤에만 관측한다. 벽 밖 이동은 같은 위치에 머물고 실패 수집도 행동 예산을 소비한다. 목표3개 수집 또는 예산 소진 시 종료한다.", "",
        "| 조건 | 훈련 번식 점수 R | 번식 기준 S |", "|---|---|---|",
        "| R0S0 | 원래 수집+1·매 행동−0.01 | 평균 보상 |",
        "| R1S0 | 원래 보상+새 최소거리 감소/60 | 평균 훈련 점수 |",
        "| R0S1 | 원래 보상 | 훈련 완료율 우선·평균 보상 차순위 |",
        "| R1S1 | 원래 보상+진행 점수 | 훈련 완료율 우선·평균 훈련 점수 차순위 |", "",
        "진행 점수는 현재 목표별 과거 최소거리보다 새로 가까워진 이동에만 지급한다. 왕복·재방문·벽·실패 수집·목표 초기화는 중복 지급하지 않는다. 목표당 최대0.1, 에피소드당 최대0.3이다. 이 정보는 정책 입력에 들어가지 않으며 최종 평가 보상에도 들어가지 않는다. S1의 순서는 동률을 유지한 서열값으로 기존 유전 연산에 전달했다.", "",
        "집단20·100회차·개체당 매회 훈련24개를 유지했다. 초기4모듈·최대9모듈이며 양수 순환 시딩과 ema/latch를 허용한다. 상위20% 엘리트, 크기3 토너먼트, 교차 확률0.65·변이 확률0.5 및 기존 변이 종류를 유지했다. 동일 초기화와 짝 시드를 사용했다. 번식하지 않는 같은 예산의 random은 네 조건에서 동일함을 사전에 검사해 시드당 하나만 공통 대조로 실행했다. 조건별 독립 random 네 집단으로 세지 않는다.", "",
        "## 2. 평가와 사전 판정", "",
        "훈련756·검증252·테스트504개 목록을 사용한다. 검증은0,9,19,…,99회차의 모든20개체에 난수3반복을 적용한다. 후보는 모든 조건에서 동일한 검증 완료율·원래 보상 순서로 고르며 동률이면 먼저 본 후보를 유지한다. 전체 탐색과 감사가 끝나면 모든 후보와 blind를 잠근 다음 테스트를 연다. best·initial·final 각각504개×3반복이며 같은 실행의 같은 유전체는 역할 간 캐시를 공유한다.", "",
        "본 시드401~420의20짝이 통계 표본 단위다. 세 처리−R0S0 대비를 한 Holm 보정군으로 등록했다. 평균 차이10%포인트 이상, 양수 차이15/20 이상, 정확 단측 부호 검정의 Holm 통과가 진행 기준이다. 부호 검정은 양수 차이 확률을 검정하며 평균 차이 검정이 아니다. 0은 양수로 세지 않았다. 평균 차이 구간은 동일한10,000회 짝 시드 bootstrap의 개별 기술적95% 구간이며 동시 구간이 아니다.", "",
        "확인 우선순위는 R1S0→R0S1→R1S1이고, 관측 평균이 가장 큰 조건으로 바꾸지 않는다. 지정된 한 조건과 R0S0 및 공통 random을 새 시드501~520에서 비교하며, 평균10%포인트·양수15/20·단측 부호 p≤0.05를 모두 요구한다. 실패 뒤 다른 조건 확인이나 시드 추가는 하지 않는다. 상호작용·random/blind 대비·행동 진단은 탐색적 보조 분석이다.", "",
        "이 benchmark는 선행 실험에서 이미 관측했다. 새 탐색 시드를 쓰더라도 새로운 환경 일반화 검증은 아니다. 초기 초안의10시드·두 대비는 성능을 보기 전20시드·세 대비로 확정했고 사전 계획과 개발 기록에 이유를 남겼다.", "",
        "## 3. 본 비교 결과", "",
        "완료율 단위는%, 표준편차는20개 탐색 시드 사이의%포인트다.", "",
        "| 조건·방법 | 평균 완료율 | 중앙값 | 표준편차 | 최소~최대 | 80% 이상 시드 |", "|---|---:|---:|---:|---:|---:|"]
    def table(summary):
        for name, row in summary["descriptions"].items():
            lines.append(f"| {name} | {pct(row['mean'])} | {pct(row['median'])} | {pct(row['sd'])} | {pct(row['minimum'])}~{pct(row['maximum'])} | {row['at_least_80_percent_seeds']}/20 |")
    table(main)
    lines += ["", "| 처리−R0S0 | 평균 차이 pp | 중앙값 pp | 기술적95% 구간 pp | 양수/음수/0 | 단측 p | Holm | 진행 |", "|---|---:|---:|---|---|---:|---|---|"]
    for name, row in main["decision"]["primary"].items():
        lines.append(f"| {name} | {pct(row['mean'])} | {pct(row['median'])} | [{pct(row['ci95'][0])}, {pct(row['ci95'][1])}] | {row['positive_seeds']}/{row['negative_seeds']}/{row['zero_seeds']} | {row['sign_p']:.8f} | {row['holm_pass']} | {row['progress_gate']} |")
    if selected:
        confirm = summaries["confirmation"]
        row = confirm["decision"]["confirmation"]
        lines += ["", f"사전 우선순위에 따른 확인 대상은 {selected}이다.", "", "## 4. 새 시드 확인", "",
                  "| 조건·방법 | 평균 완료율 | 중앙값 | 표준편차 | 최소~최대 | 80% 이상 시드 |", "|---|---:|---:|---:|---:|---:|"]
        table(confirm)
        lines += ["", f"{selected}−R0S0 평균 차이 **{pct(row['mean'])}%포인트**, 중앙값 {pct(row['median'])}%포인트, 기술적95% 구간 [{pct(row['ci95'][0])}, {pct(row['ci95'][1])}]%포인트. 양수/음수/0={row['positive_seeds']}/{row['negative_seeds']}/{row['zero_seeds']}, 단측 p={row['sign_p']:.8f}. 사전 확인 기준 **{'충족' if row['progress_gate'] else '미충족'}**.", "",
                  "본 비교와 확인을 합쳐 실패를 만회하는 판정을 하지 않았다. 다른 후보로 바꾸지 않았다."]
    else:
        lines += ["", "세 처리 모두 사전 진행 기준을 충족하지 않아 확인을 실행하지 않았다. 효과를 확인하지 못한 것을 동등성이나 일반적인 능력 부재로 해석하지 않는다."]
    lines += ["", "## 5. 시드별 원 수치와 보조 분석", ""]
    for phase, summary in summaries.items():
        config = read(DATA / phase / "protocol.json")
        names = list(summary["descriptions"])
        lines += [f"### {phase}: 완료율(%)", "", "| 시드 | " + " | ".join(names) + " |", "|---|" + "---:|"*len(names)]
        for i, seed in enumerate(config["seeds"]):
            lines.append(f"| {seed} | " + " | ".join(pct(summary["descriptions"][n]["rates"][i]) for n in names) + " |")
        lines += ["", "| 조건·방법 | initial 평균% | final 평균% | best 평균% | 초기 짝 프로브% |", "|---|---:|---:|---:|---:|"]
        for name, group in completions[phase]["groups"].items():
            lines.append(f"| {name} | {pct(group['initial']['success_rate']['mean'])} | {pct(group['final']['success_rate']['mean'])} | {pct(group['best']['success_rate']['mean'])} | {pct(group['probe_rate']['mean'])} |")
        lines += ["", "| 조건·방법 | best 원래 평균 보상 | 평균 수집 수 | 평균 실패 수집 | 평균 벽 충돌 |", "|---|---:|---:|---:|---:|"]
        for name, group in completions[phase]["groups"].items():
            metrics = group["best"]
            lines.append("| " + name + " | " + " | ".join(f"{metrics[key]['mean']:.4f}" for key in ("mean_reward", "mean_collected", "mean_failed_collect", "mean_walls")) + " |")
        lines += ["", "| 조건·방법 | 첫 행동 정답% | 첫 오답 뒤 완수% | 목표 전환 다음 정답% | 실패 중 끝6행동 동일% |", "|---|---:|---:|---:|---:|"]
        for name in names:
            condition, method = name.split("/")
            rows = [r["trajectory"] for r in summary["rows"] if r["condition"] == condition and r["method"] == method]
            def ratio(a, b):
                denominator = sum(r[b] for r in rows)
                return pct(sum(r[a] for r in rows)/denominator) if denominator else "해당 없음"
            lines.append(f"| {name} | {ratio('first_correct','episodes')} | {ratio('recovered_after_first_wrong','first_wrong')} | {ratio('goal_change_correct','goal_changes')} | {ratio('repeated_last_six_failures','failures')} |")
        lines += ["", "행동 표는 실제 방문 상태를 합친 기술 통계다. 독립 표본 수를 에피소드·목표 전환 수로 늘려 검정하지 않았다. 초기 상태 좌우 짝 프로브는105짝/실행이고 과정 전체 성공·오답 회복과 다른 능력을 측정한다. 각 분모와 정수 건수는 summary의 trajectory에 보존했다.", ""]
    inter = main["decision"]["interaction_exploratory"]
    lines += [f"탐색적 상호작용 R1S1−R1S0−R0S1+R0S0 평균={pct(inter['mean'])}%포인트, 구간=[{pct(inter['ci95'][0])}, {pct(inter['ci95'][1])}]%포인트. 주 판정을 대체하지 않는다.", "",
              "기준 정책의 테스트 완료율: " + ", ".join(f"{k} {pct(v['success_rate'])}%" for k,v in main["baselines"].items()) + ". blind는 검증252개에서140고정 패턴 중 선발했다. uniform은 난수20반복, coordinate는 수작업 정답 규칙이다.", "",
              "번식 기준의 선호 충돌·순서 차이·엘리트 집합 차이·진행 보상에 의한 순서 변화의 회차 합계는 완료 집계의 ranking_history_totals에 있다. 이를 독립적인 효과 검정이나 원인 확정으로 해석하지 않는다.", "",
              "## 6. 계산량·감사·오류 기록", "",
              "| 단계 | 실행 수 | 훈련 논리 슬롯 | 검증 논리 슬롯 | 실제 탐색 호출 | 테스트 논리 슬롯 | 실제 테스트 호출 | 미확정 중단 상한 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for phase, completion in completions.items():
        c = completion["counts"]
        lines.append(f"| {phase} | {completion['jobs']} | {c['search_training_episode_slots']:,} | {c['search_validation_episode_slots']:,} | {c['search_known_completed_episode_calls']:,} | {c['test_logical_episode_slots']:,} | {c['test_known_completed_episode_calls']:,} | {sum(v for k,v in c.items() if k.endswith('unknown_interrupted_calls_upper_bound')):,} |")
    lines += ["", "공식 예비10실행24,000훈련 호출, 추가 비용 표본18,336훈련 목록 호출을 별도 기록했다. 본 blind 검증35,280·기준 정책 테스트11,088호출은 위 후보 표와 별도다. 확인에서는 동일 baseline/blind를 새로 평가하지 않고 본 기록을 고정해 공유한다. 프로브는 실행당210단일 행동이며 에피소드 호출에 합치지 않는다. 역할 캐시는 동일 실행·동일 유전체의 중복 테스트를 줄이며 검증 선발 횟수나 탐색 예산을 줄이지 않는다.", "",
              "전체 원자료의 행동으로 환경·보상·진행 점수·종료를 재생하고 모든 후보 점수·선발·유전 연산·난수·ID와 테스트 캐시를 감사했다. 모든 정책 네트워크의 전 행동을 별도로 다시 계산한 감사는 아니다. 독립적인 scalar/batch 정합성 검사와 고정 소스 지문을 함께 사용했다. 자동 검사는 모든 버그의 부재 또는 독립 과학 검토를 뜻하지 않는다.", "",
              "성능 실행 전에 R1 합산을 정수1/300단위로 확정하고 두 수치 경계 검사를 추가했다. 개발46+99검사 기록과 최종48+99검사 기록을 모두 보존했다. 예비 이후 핵심 소스·수치 계획·게이트 변경은 없었다. GitHub Linux에서는 원본48검사 중 혼합 훈련/검증 fixture의 공통 random 전체 사전 일치 검사1개가 실패했다. 본 실험은 검증 전용 선발이므로 별도 tools/check_q1005_portable.py에서 원본47검사와 검증 전용·본 분기 검사2개를 실행해49개 통과를 확인했다. 원래 검사와 실패 기록은 보존했고 세부 원인은 [CI 기록](../management/q1_005_ci_portability.json)에 있다. 실제 본100실행의 best·initial·final300역할은 전부 등록 검증 회차·검증 전용 지표임을 [후보 출처 검사](../management/q1_005_candidate_sources.json)로 추가 확인했다. 실행 실패·제외는 완료 집계의 attempt 원장 및 failures 파일 존재 여부로 확인했다. 실패를 기록하지 않았다는 사실만으로 일반적인 무오류를 주장하지 않는다.", "",
              "## 7. 해석과 다음 질문", "",
              "이 비교는 현재 A0·작은 이동 수집 과제·고정 탐색 예산에서 적합도 신호의 효과를 평가한다. 개별 고성능과 여러 시드에서의 안정적인 발견을 구분해야 한다. 일반 지능의 성립이나 진화 접근 전체의 불가능성, 더 큰 구조의 필요성을 이 결과 하나로 결정할 수 없다.", "",
              "다음 예정 질문은 Q1-006의 고정 구조 가중치 탐색과 구조 변경 허용의 비교다. 보상·번식·예산까지 동시에 바꾸지 않고, 동일 시작 템플릿·무효 변이·실제 비용·확인 기준을 먼저 등록한다. 이번 결과를 보고 가장 큰 모델이나 가장 유리한 시드로 바로 넘어가지 않는다.", "",
              "## 8. 재현과 보존", "",
              f"소스34파일 지문 `{protocol['source_digest']}`. 실제 런타임은 Python {protocol['python']}·NumPy {protocol['numpy']}·{protocol['platform']}다. 수치 계획·런타임·데이터·게이트 지문은 protocol.json에 있다. [실행 준비](../management/Q1-005_실행준비검증.md)에147검사와 예비·비용 표본을 기록했다. 소스 ZIP도 원자료 ZIP 안에 포함한다.", "",
              "작업 디렉터리 `studies/interaction/`에서 사용한 명령:", "", "```powershell",
              "python -B -u -X utf8 -m evosim_q1005_v01.run pilot --gate ../../docs/management/q1_005_execution_gate_final.json --workers 2",
              "python -B -u -X utf8 -m evosim_q1005_v01.run main --gate ../../docs/management/q1_005_execution_gate_final.json --pilot-report results_q1005_v01/pilot/cost_report.json --workers 2"]
    if selected:
        lines.append("python -B -u -X utf8 -m evosim_q1005_v01.run confirmation --gate ../../docs/management/q1_005_execution_gate_final.json --workers 2")
    lines += ["```", "", "완료 자료를 읽으려고 실험을 재실행할 필요는 없다. 저장소 루트의 `tools/collect_q1005.py`는 완료 자료만 읽고 `tools/restore_q1005.py 다운로드ZIP --verify-only`는 파일별 해시를 검증한다. 기존 파일이 다르면 덮어쓰지 않는다. 원자료 다운로드 주소·SHA-256은 evidence 목록에 둔다.", "",
              "## 수행·작성·검토 기록", "",
              "- 연구 방향·실험 요청자: 사용자. Q1 원인 분리 계획에 따라 계속 진행하도록 요청함.",
              "- 실험 수행자: Codex 에이전트. 로컬 Python 구현·검사·예비·본 비교·조건부 확인 여부 판정·원자료 감사·집계·보관을 수행함.",
              "- 실험 수행일: 2026-10-16 한국 시간.", "- 보고서 작성자: Codex 에이전트.", "- 문서 작성일: 2026-10-16 한국 시간.",
              "- 모델·버전: 실행 당시 정확한 호출 모델 ID 미기록.", "- 독립 검토 상태: 독립 검토 기록 없음. 자동 검사·기록 감사와 구분함.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(REPORT)


if __name__ == "__main__":
    main()
