# 공통 기반 v0.2 — 실제 관측 열 저장

평가 커널과 원자 Store는 [v0.1](../evosim_runtime_v01/README.md)을 수정 없이 재사용한다. 새 `storage.TableCodec`은 환경 재생 없이 기록된 값 자체를 복원한다. 공통 에피소드·반복 순서·유전체·행동열은 공유하고, 관측 열의 정수/불리언은 범위와 타입을 검사해 int8/int16/bool 배열과 base64로 저장한다. 부호 있는 다중 바이트 정수는 little endian으로 명시한다.

범위를 넘는 정수, 부동소수점, 섞인 타입, 임의의 새 값은 원래 JSON 열로 보존한다. 일부 행에만 있는 필드는 위치 목록을 저장한다. 필수 환경 필드가 없는 기록도 원문으로 유지한다. 유효하지 않은 행동·에피소드 ID·이상 보상을 고쳐 쓰지 않는다. 따라서 로그의 의미가 맞는지는 복원 뒤 별도 감사로 확인한다.

```python
from evosim_runtime_v02.storage import Store, TableCodec
store = Store(TableCodec(), level=9)
store.write('new_record.json.gz', observed_record)
restored = store.read('new_record.json.gz')
```

schema·내용 해시·원본 전체 JSON 해시를 검증한다. 관측값 자체를 저장하므로 복원에 환경 코드를 실행할 필요가 없다. 실험 자체의 코드/프로토콜·난수·분할 해시는 실행 원장에 계속 보존해야 한다. 누락된 입력·네트워크 출력을 이 형식이 새로 만들어 주지는 않는다.

통합 기술 API는 [experiment_runtime_v01.py](../experiment_runtime_v01.py)의 `ExperimentRuntime`이다.

| 프로필 | 평가 | 기록 | 용도 |
|---|---|---|---|
| `reference` | 기준 후보별 평가 | 원래 JSON·gzip3 | 비교·재현 기준 |
| `balanced` 기본 | 지원 구조 집단 평가·그 외 기준 콜백 | 실제 관측 열·gzip9 | 실행 속도·복원·용량의 절충 |
| `archive` | 동일 집단 평가 | v0.1 행동 재생·실제 예외값·gzip3 | 더 작은 장기 보관, 처리 비용 증가 |

실제 지원 범위는 현재 이동·수집의 훈련 전용 기술 어댑터다. 본 실행기는 아직 통합하지 않았다. 완료 소스와 측정은 버전별로 보존한다. [Q1-011 결과](../../../docs/reports/Q1-011_실험기반_최적화검증.md)에서 모든 측정값과 적용 한계를 확인한다.
