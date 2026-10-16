# Q1-005 보상·번식 기준 비교

실행 소스는34개 Python 파일 지문으로 고정한다. 이 패키지14파일과 기존 `evosim_interaction_v02/`의 동결20파일을 함께 검사한다. 기존 의존성을 수정하지 않는다. 연구 디렉터리 `studies/interaction/`에서 실행한다.

공식 설계는 저장소의 `docs/protocols/Q1-005_보상과번식_실험계획.md`, 수치는 `q1_005_v01_plan.json`, 실제 검사·예비·비용·해시는 `docs/management/Q1-005_실행준비검증.md`에 있다.

```powershell
python -B -X utf8 -m unittest discover -s evosim_q1005_v01/tests -v
python -B -u -X utf8 -m evosim_q1005_v01.run pilot --gate ../../docs/management/q1_005_execution_gate_final.json --workers 2
python -B -u -X utf8 -m evosim_q1005_v01.run main --gate ../../docs/management/q1_005_execution_gate_final.json --pilot-report results_q1005_v01/pilot/cost_report.json --workers 2
```

위는 실제 실행에 사용한 명령이다. 완료 자료를 읽기 위해 다시 실행할 필요는 없다. 완료 폴더에는 다른 코드·설정·환경을 이어 쓰지 않는다. 결과는 `results_q1005_v01/`의 단계별 폴더에 보존한다. 후보·baseline 선발 전에는 테스트를 열지 않으며 R1 훈련 점수는 공통 검증/시험 지표와 구분한다.
