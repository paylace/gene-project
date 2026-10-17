# Q1-007 v0.2 기술 어댑터

Q1-007 v0.1의 유전체·초기 분포·유전 연산·지표 정의를 수정 없이 재사용하고 [공통 기반](../evosim_runtime_v01/README.md)의 집단 평가·저장을 연결한다. 과거 코드를 실행 중 monkeypatch하거나 여러 실험 폴더를 다시 복사하지 않는다. 동결된 의존 코드도 provenance에 포함한다.

현재 범위는 **훈련 전용 기술 검증**이다. `evaluation.evaluate_population`은 훈련 분할 이외 입력을 거부한다. `fixture.run_fixture`는 최대20회차의 기술 탐색으로 선발·유전체·계보·중단/재개·원장을 확인한다. 실제 본 프로토콜의 검증 간격·후보 잠금·테스트 실행을 구현한 본 실행기로 오해하지 않는다. Q1-007 본·확인 진입점은 없다.

## API

```python
from evosim_q1007_v02.evaluation import evaluate_population

results = evaluate_population(
    genomes, training_episodes, seed, candidate_purposes,
    training=True, repeats=1, backend='auto')
# 후보 순서대로 (실제 지표, 전체 에피소드 기록)
```

`backend='fixed'`는 지원 그래프를 강제하며 미지원 구조는 거부한다. `auto`는 미지원 구조를 기존 평가 콜백으로 보내고 `reference`는 모든 후보를 기존 경로로 평가한다. 기존 경로가 정의하지 않는 새 구조는 추가 정의와 검사가 필요하다. 목적 문자열과 훈련 점수 모드를 혼동하면 거부한다.

기술 fixture는 회차별 시작·훈련 평가·전체 평가·커밋 상태와 호출 상한을 저장한다. 원자료의 내용 해시와 코드/분할/설정 식별자를 확인한 뒤 재개한다. 완료 회차가 바뀌었거나 실행 코드가 달라지면 거부한다. 비교 결과는 당시 코드의 정합성 근거이며 일반적 탐색 성능 향상을 의미하지 않는다.

앞으로 본에 적용할 때에는 별도 구현·비용 게이트·과학적 프로토콜을 잠근다. 이번 기술 벤치마크가 이전 비용 미통과를 자동으로 통과시키지 않는다.
