# 환경 상호작용 연구 — Q1 v0.1 완료

연구 목표와 계획은 [연구 목표와 진행 계획](../연구_목표와_진행계획.md)에서 관리한다.

일곱 칸 이동·수집 환경, 네 입력 네트워크, 교차·변이·동결 평가·탐색·후보 선발·최종 평가를 구현하고 본 시드 10쌍의 실험을 완료했다. 기존 `evosim_research/`의 코드나 결과는 수정하지 않았다.

환경·기준 정책은 표준 라이브러리를 사용하고, 네트워크 묶음 평가와 분석은 NumPy를 사용한다. 실행한 Python·NumPy 버전은 `results_v01/protocol.json`에 보존했다. 상위 작업 폴더에서 다음 명령으로 21개 테스트를 재현한다.

```powershell
& 'C:\Users\thoma\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s evosim_interaction/tests -v
```

- `environment.py`: 행동에 따라 위치·목표를 갱신하며 종료 후 호출과 잘못된 행동을 거부한다.
- `datasets.py`: 목표 목록을 단위로 훈련 756개·선발 252개·테스트 504개를 나눈다.
- `policies.py`: 좌표 비교 정책, 관측 없는 고정 순서, 균등 무작위 정책을 제공한다.
- `rollout.py`: 정책에 네 관측값만 전달하고 행동·보상·전후 상태를 기록한다.
- `tests/test_foundation.py`: 전체 1,512조합의 규칙 정책, 경계·종료·보상, 분할, 난수 격리, 에피소드 초기화를 검증한다.

전체 조합 검사는 수작업 규칙 정책에 대한 환경 검증이다. 별도의 본 실험에서는 무관측 후보 선발과 진화 개체의 최종 테스트를 완료했다. 진화 후보의 세 목표 완료 성공률은 평균 0.1786%, 같은 예산의 무작위 개체 탐색은 0.1521%였다. 사전 긍정 기준을 충족하지 못했으며, 자세한 해석과 사후 진단은 계획 파일의 8절에 기록했다.

추가 구현:

- `network.py`: 네 입력·세 출력, 실행별 모듈 ID, 순환 상태, 교차·변이, 개별·묶음 순전파.
- `evaluation.py`: 생애 학습 없는 평가, 에피소드별 기록, 관측 35쌍 검사.
- `search.py`, `run.py`: 같은 평가 예산의 두 탐색 방법, 회차별 체크포인트, 명세 고정.
- `finalize.py`: 모든 후보 고정 후 최종 평가, 행동 기록을 환경에서 재생해 검사.
- `analysis.py`: 사전 시드 단위 집계와 판정 규칙.

상위 작업 폴더에서 재현할 명령이다. 새 버전을 실행하려면 다른 출력 폴더를 사용한다. 기존 실행의 소스나 명세가 달라지면 재개를 거부한다.

```powershell
& 'C:\Users\thoma\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m evosim_interaction.run pilot --output evosim_interaction/pilot_v01 --workers 2
& 'C:\Users\thoma\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m evosim_interaction.run search --output evosim_interaction/results_v01 --workers 2
& 'C:\Users\thoma\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m evosim_interaction.finalize evosim_interaction/results_v01
& 'C:\Users\thoma\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' audit_interaction.py evosim_interaction/results_v01
& 'C:\Users\thoma\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' diagnose_interaction.py evosim_interaction/results_v01
& 'C:\Users\thoma\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' probe_action_bottleneck.py evosim_interaction/results_v01
& 'C:\Users\thoma\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' summarize_interaction.py evosim_interaction/results_v01
```

마지막 네 스크립트는 본 실험의 해시가 고정된 소스와 분리해 작업 폴더에 두었다. 두 원인 진단은 본 결과를 확인한 뒤 설계한 탐색적 검사이며 재훈련·후보 재선발이 없다. 그림 생성에는 설치된 ReportLab 및 Windows 맑은 고딕 글꼴을 사용한다.
