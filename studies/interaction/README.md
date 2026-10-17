# 환경 상호작용 연구 실행 안내

이 폴더를 작업 디렉터리로 사용한다. 동결 v0.1은 `evosim_interaction/`, Q1-004의 별도 구현은 `evosim_interaction_v02/`, 새 결과는 `results_v02/`에 있다. 감사·사후 진단 도구도 연구별 버전으로 구분한다.

공식 보고서는 [Q1 보고서 목록](../../docs/reports/README.md), 연구 목적은 [목적 문서](../../docs/01_연구목적.md), 현재 상태는 [현황](../../docs/03_연구현황.md)에서 확인한다.

## Q1-011 — 실험 기반과 로그 최적화

[결과](../../docs/reports/Q1-011_실험기반_최적화검증.md), [공통 기반 안내](evosim_runtime_v02/README.md), [단일 프로필 API](experiment_runtime_v01.py)를 확인한다. `balanced`는 지원 구조 집단 평가와 실제 관측 열 gzip9를, `archive`는 더 작은 재생형을 사용한다. 기존 실험은 변경하지 않았고 기술 어댑터는 훈련 분할만 허용한다. 본 실행기 통합·전체 비용 검증은 미실행이다.

## Q1-007 v0.1 — 준비 완료·비용 미통과로 본 보류

[준비 결과](../../docs/reports/Q1-007_탐색예산_비교.md)에 신규20검사·훈련 전용6예비76,800호출·추가 비용45,360호출·감사를 기록했다. 소스는 `evosim_q1007_v01/`, 출력은 `results_q1007_v01/`다. 시간의2배 상한5.44시간·비압축 JSON44.77GiB가 계획 기준을 넘어서 본/확인 호출0이다. 소스·예비·비용 판정을 동결 보존하고 새 버전에서 저장·평가 효율화를 설계한다. 압축 저장량과 비압축 논리량을 구분하며 낮은 개체 성능으로 해석하지 않는다.

## Q1-006 v0.2 — 고정 구조·구조 탐색 비교 완료

[실행 계획](../../docs/protocols/Q1-006_v02_고정구조와구조탐색_실험계획.md)과 [준비 검증](../../docs/management/Q1-006_v02_실행준비검증.md)에 같은 초기4모듈 tanh의 F0/F1·40짝 시드·공통 무작위 대조·사전 판정을 등록했다. 신규20+기존173=193검사와 훈련 전용8예비를 통과했다. [공식 결과](../../docs/reports/Q1-006_고정구조와구조탐색_비교.md)에 본120작업·F0 우세 기준 미충족·확인 미실행을 기록했다. 소스는 `evosim_q1006_v02/`, 결과는 `results_q1006_v02/`다. v0.1의 예비 구성 오류·원자료는 별도 폴더에 보존했으며 덮어쓰지 않는다.

## Q1-009 v0.1 — 초기 반응·고착 진단 완료

[공식 결과](../../docs/reports/Q1-009_초기반응과고착_요인분리.md), [실행 계획](../../docs/protocols/Q1-009_초기반응과고착_요인분리계획.md), [완료 요약](results_q1009_v01/main/summary.json)에 초기500원형×4조건과 고정 대조5개를 기록했다. 학습·진화 없는 진단이며 소스는 `evosim_q1009_v01/`, 출력은 `results_q1009_v01/`이다. 원자료는 [복원 안내](../../evidence/README.md)와 [목록](../../evidence/q1_009_v01.json)을 따른다. 결과 열람을 위해 다시 실행할 필요는 없다.

다음 Q1-006의 [템플릿 준비 검사](../../docs/management/Q1-006_템플릿_준비검사.md) 원자료는 `q1006_preflight_v01/template_check.json.gz`에 별도로 둔다. 수작업 가중치의 표현 가능성 검사이며 Q1-009 원자료 ZIP이나 진화 성과에 합산하지 않는다. 그 뒤의 Q1-006 본 비교는 위 절에 완료 결과를 연결했다.

## Q1-005 v0.1 — 보상·번식 비교 완료

구현은 [evosim_q1005_v01](evosim_q1005_v01/README.md), 개별 설계는 [계획서](../../docs/protocols/Q1-005_보상과번식_실험계획.md), 실행 전147검사·예비와 CI 이식성 문제는 [준비 검증](../../docs/management/Q1-005_실행준비검증.md)에 있다. [공식 결과](../../docs/reports/Q1-005_보상과번식_분리비교.md)에 본100실행·세 대비 기준 미충족·확인 미실행을 기록했다. 출력은 `results_q1005_v01/`이며 기존 실행과 섞지 않는다. 원본 검사의 혼합-purpose fixture 문제를 기록한 별도 CI 검사는 루트의 `python -B -X utf8 tools/check_q1005_portable.py`로 실행한다. 동결 소스와 기존 게이트는 보존한다.

## Q1-004 v0.2 — 완료

[공식 보고서](../../docs/reports/Q1-004_구조제약_재진화비교.md)에 본60·확인40실행, 확인 실패와 사후 기술 점검을 기록했다. [본 집계](results_v02/main/summary.json)와 [확인 집계](results_v02/confirmation/summary.json)는 구분해서 읽는다. 결과를 다시 읽기 위해 실험을 재실행할 필요는 없다.

```powershell
python -B -u -X utf8 -m evosim_interaction_v02.run pilot --gate ../../docs/management/q1_004_execution_gate.json --workers 2
python -B -u -X utf8 -m evosim_interaction_v02.run main --gate ../../docs/management/q1_004_execution_gate.json --pilot-report results_v02/pilot/cost_report.json --workers 2
python -B -u -X utf8 -m evosim_interaction_v02.run confirmation --gate ../../docs/management/q1_004_execution_gate.json --workers 2
```

위는 실제 사용 명령이다. 고정된 소스·프로토콜·실행 환경을 요구한다. 다른 코드·환경을 완료 폴더에 이어 쓰지 않는다. 구현 게이트를 다시 실행할 때 원래 게이트 파일을 덮어쓰지 말고 별도 경로를 사용한다. 원자료 다운로드·해시 복원은 [원자료 안내](../../evidence/README.md)를 따른다.

## v0.1 검증과 결과 위치

```powershell
python -m unittest discover -s evosim_interaction/tests -v
```

- 명세·데이터·후보 고정: [results_v01](evosim_interaction/results_v01/)
- 주 집계: [summary.json](evosim_interaction/results_v01/summary.json)
- 원자료 감사: [audit.json](evosim_interaction/results_v01/audit.json)
- 표현·기억 진단: [exploratory_diagnostics](evosim_interaction/results_v01/exploratory_diagnostics/)
- 행동 병목 진단: [action_bottleneck](evosim_interaction/results_v01/action_bottleneck/)

## v0.1 재현 명령

```powershell
python -m evosim_interaction.run pilot --output evosim_interaction/pilot_v01 --workers 2
python -m evosim_interaction.run search --output evosim_interaction/results_v01 --workers 2
python -m evosim_interaction.finalize evosim_interaction/results_v01
python audit_interaction.py evosim_interaction/results_v01
python diagnose_interaction.py evosim_interaction/results_v01
python probe_action_bottleneck.py evosim_interaction/results_v01
python summarize_interaction.py evosim_interaction/results_v01
```

본 결과를 읽기 위해 위 실험을 다시 실행할 필요는 없다. 소스·명세가 달라지면 기존 결과와 같은 실행으로 이어 붙이지 않으며 새 출력 폴더를 사용한다. 두 진단은 사후 탐색이고 후보 재선발·재훈련은 없다.

정확한 Python·NumPy 버전은 고정 프로토콜에 있다. 그림 도구만 ReportLab과 Windows 맑은 고딕에 의존한다. 임시 폴더 권한 문제가 있으면 원래 테스트 코드를 바꾸지 않고 허용된 작업 폴더 안의 임시 디렉터리로 실행 환경을 지정한다.
