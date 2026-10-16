# Q1-004 v0.2 기반 구현

현재 불변 설정·유전 연산에 이어 동결 평가·관찰기·조건별 탐색·원자 저장/재개·후보 잠금·분석/감사를 구현했다. 기존 `evosim_interaction/` 완료 소스는 수정하지 않는다. 소스 변경 시 기존 실행의 재개를 거부한다.

`Settings("A0"/"A1"/"A2")`를 모든 공개 생성·변이·교차·정책 API에 전달한다. 유전체의 `to_dict()`와 `digest`는 기존 유전자 표현을 유지하여 A0와 직접 비교할 수 있다. 저장·복원에는 조건/설정 해시를 포함한 `Genome.export()` / `Genome.restore(snapshot, settings)`를 사용한다. `from_dict(data, settings)`는 엄격한 유전자 디코더이며 조건 위반을 수리하지 않는다. 조건을 보존하는 저장본으로는 export 형식을 사용한다.

`settings.to_dict()`의 `baseline_parameters`는 v0.1 제안 모수이며 실제 허용 구조는 `constraints`에 따로 기록한다. 예를 들어 A2도 제안은 기존 6연산에서 뽑지만 실제 ema/latch는 허용하지 않는다. 제안 가중치를 가능한 연산에 재분배하지 않는다.

저장소 루트에서 구현 검사를 실행한다.

```powershell
python -B -X utf8 tools/check_q1_core.py
```

검사 JSON/로그는 기본적으로 업로드 제외된 `dist/q1_core_verification/`에 생긴다. W1/W2 당시 기록은 루트 `docs/management/q1_004_core_checks.json`, 현재52개 패키지 검사+기존47개 게이트는 `docs/management/q1_004_execution_gate.json`이다. 실행 준비·예비·비용·명령은 `docs/management/Q1-004_실행준비검증.md`에 기록한다. 검사와 실제 성능 결과를 구분한다.

본 패키지는 연구 디렉터리에서 `python -m evosim_interaction_v02.run pilot/main/confirmation`으로 실행한다. `--gate`가 필요하며 main은 matching-source 예비 비용 기록도 요구한다. confirmation은 본 비교의 사전 진행 기준이 통과한 조건만 허용한다. 예비에는 validation/test 평가가 없다. 본 테스트는 전 실행·blind 후보 잠금 뒤에만 열며 재평가 캐시는 같은 조건·시드·고정 프로토콜 내부에서만 사용한다.
