# 환경 상호작용 연구 실행 안내

이 폴더를 작업 디렉터리로 사용한다. Python 패키지와 실행 자료는 `evosim_interaction/`, 감사·사후 진단 도구는 이 폴더에 있다.

공식 보고서는 [Q1 보고서 목록](../../docs/reports/README.md), 연구 목적은 [목적 문서](../../docs/01_연구목적.md), 현재 상태는 [현황](../../docs/03_연구현황.md)에서 확인한다.

## 검증과 결과 위치

```powershell
python -m unittest discover -s evosim_interaction/tests -v
```

- 명세·데이터·후보 고정: [results_v01](evosim_interaction/results_v01/)
- 주 집계: [summary.json](evosim_interaction/results_v01/summary.json)
- 원자료 감사: [audit.json](evosim_interaction/results_v01/audit.json)
- 표현·기억 진단: [exploratory_diagnostics](evosim_interaction/results_v01/exploratory_diagnostics/)
- 행동 병목 진단: [action_bottleneck](evosim_interaction/results_v01/action_bottleneck/)

## 재현 명령

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
