# Q1-009 초기 반응·고착 진단

동결 A0 원형에 좌표 중심화(C)와 출력 자기순환 제거(N)를 따로 적용한다. 초기 유전체만 검사하며 학습·진화·후보 선발은 없다. 기존 동결 소스를 수정하지 않는다.

설계는 `docs/protocols/Q1-009_초기반응과고착_요인분리계획.md`, 수치는 같은 폴더의 `q1_009_v01_plan.json`에 있다. 전체 소스·수치 계획·런타임·게이트가 일치해야 재개할 수 있다. 원자료는 `../results_q1009_v01/`에 보존한다.

작업 디렉터리 `studies/interaction/`:

```powershell
python -B -u -X utf8 -m evosim_q1009_v01.run pilot --gate ../../docs/management/q1_009_execution_gate_final.json --workers 2
python -B -u -X utf8 -m evosim_q1009_v01.run main --gate ../../docs/management/q1_009_execution_gate_final.json --workers 2
```

검사는 이 작업 디렉터리에서 `python -B -X utf8 -m unittest discover -s evosim_q1009_v01/tests -v`다. 자동 검사는 독립 과학 검토가 아니다. 완료한 출력 파일은 덮어쓰지 않는다.
