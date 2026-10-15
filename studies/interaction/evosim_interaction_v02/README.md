# Q1-004 v0.2 기반 구현

현재 W1 불변 설정·엄격한 저장/복원, W2 조건별 초기화·변이·교차·한 회차 번식을 구현했다. 기존 `evosim_interaction/` 완료 소스는 수정하지 않는다. 기본 scalar/batch 정책 계산도 A0 대조용으로 분리했으나 W3 관찰기·전체 평가기와 W4 탐색/재개 실행기는 아직 없다.

`Settings("A0"/"A1"/"A2")`를 모든 공개 생성·변이·교차·정책 API에 전달한다. 유전체의 `to_dict()`와 `digest`는 기존 유전자 표현을 유지하여 A0와 직접 비교할 수 있다. 저장·복원에는 조건/설정 해시를 포함한 `Genome.export()` / `Genome.restore(snapshot, settings)`를 사용한다. `from_dict(data, settings)`는 엄격한 유전자 디코더이며 조건 위반을 수리하지 않는다. 조건을 보존하는 저장본으로는 export 형식을 사용한다.

`settings.to_dict()`의 `baseline_parameters`는 v0.1 제안 모수이며 실제 허용 구조는 `constraints`에 따로 기록한다. 예를 들어 A2도 제안은 기존 6연산에서 뽑지만 실제 ema/latch는 허용하지 않는다. 제안 가중치를 가능한 연산에 재분배하지 않는다.

저장소 루트에서 구현 검사를 실행한다.

```powershell
python -B -X utf8 tools/check_q1_core.py
```

검사 JSON/로그는 기본적으로 업로드 제외된 `dist/q1_core_verification/`에 생긴다. 이번 구현의 보존된 기록은 루트 `docs/management/q1_004_core_checks.json`에 있다. 이 검사는 단위 검사와 훈련 목록 일부의 동작 대조이며 성능 실험·W6 전체 통과·프로토콜 잠금이 아니다. 본 탐색 명령은 아직 제공하지 않는다.
