# 환경 상호작용 연구 실행 안내

이 폴더를 작업 디렉터리로 사용한다. 동결 v0.1은 `evosim_interaction/`, Q1-004의 별도 구현은 `evosim_interaction_v02/`, 새 결과는 `results_v02/`에 있다. 감사·사후 진단 도구도 연구별 버전으로 구분한다.

공식 보고서는 [Q1 보고서 목록](../../docs/reports/README.md), 연구 목적은 [목적 문서](../../docs/01_연구목적.md), 현재 상태는 [현황](../../docs/03_연구현황.md)에서 확인한다.

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
