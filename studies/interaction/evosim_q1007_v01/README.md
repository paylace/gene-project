# Q1-007 탐색 자원 배분 실행기

[실행 전 계획](../../../docs/protocols/Q1-007_탐색예산_실험계획.md)의 고정4모듈·집단/회차 비교를 구현한다. Q1-006 v0.2의 신경망·유전 연산·평가기와 Q1-004의 동결 의존성을 함께 해시로 검사한다. 완료한 선행 소스는 수정하지 않는다.

20개체×200회차,40개체×200회차,무작위20개체×400회차의 세 실행에서 사전 지정한 일곱 끝점의 후보를 저장한다. 100회차 후보에 미래 후보를 넣지 않는다. 검증은 매10회차 전 개체이며0회차에는 없다. 훈련·검증·테스트의 용도를 구분하고 전체 끝점 잠금 뒤에만 테스트한다.

작업 디렉터리는 `studies/interaction/`이다. 아래 명령은 구현 검사 게이트와 사전 비용 문턱을 통과해야 한다. Python은 이 PC의 지정 실행 파일 또는 동등한 환경을 사용한다.

```powershell
python -B -X utf8 -m evosim_q1007_v01.run pilot --gate ../../docs/management/q1_007_execution_gate.json --workers 2
python -B -X utf8 -m evosim_q1007_v01.cost --gate ../../docs/management/q1_007_execution_gate.json
python -B -X utf8 -m evosim_q1007_v01.run main --gate ../../docs/management/q1_007_execution_gate.json --cost results_q1007_v01/pilot/cost_gate.json --workers 2
```

비용 게이트 실패는 성능 실패가 아니다. 실패하면 본 실행을 거부하며 원래 예비·비용 기록과 소스를 보존한다. 확인은 사전 본 판정에서 허용한 대비만 같은 run 명령의 `confirmation`으로 실행한다. 상세 원자료는 `results_q1007_v01/`에 저장되며 Git에는 요약·프로토콜만 올린다. 새 버전 없이 완료 소스를 바꾸지 않는다.
