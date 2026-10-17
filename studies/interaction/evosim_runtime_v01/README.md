# 공통 실험 기반 v0.1

환경의 집단 평가와 저장을 실험별 선발·진화 정의에서 분리한 버전 고정 기반이다. 현재 최적화 백엔드는 일곱 칸 이동·수집과 고정4모듈 tanh 그래프에 한정된다. 모든 환경에 쓸 수 있는 완성된 프레임워크로 부르지 않는다.

## 구성과 재사용 경계

| 모듈 | 책임 | 확장 시 계약 |
|---|---|---|
| `linecollect.py` | 후보×에피소드 집단 평가, 순서 보존 tanh, 동률 난수 지연 준비 | 새 구조는 `supports`와 출력 동치를 검증한 별도 백엔드. 미지원 시 기준 콜백 또는 거부 |
| `records.py` | 행동 증거로 환경 결과 벡터 복원 | 새 환경은 `schema`, `dependencies`, `restore_unique`를 제공하는 별도 codec |
| `storage.py` | 테이블 공유·관측 예외·내용 해시·원자 쓰기 | 실험별 선발이나 유전 연산을 넣지 않음. 일반 JSON은 codec 없이 사용 |
| `accounting.py` | 기준/최적화 호출·실제 행동·동률 RNG·미확정 상한 | 프로세스 단위 계수. 다중 스레드 공유 계수 용도는 미지원 |

진화·무작위 탐색·선발 지표·데이터 접근 정책은 호출자의 책임이다. Q1-007 기술 어댑터는 훈련 분할만 허용한다. 평가 API는 후보마다 고유한 목적 문자열을 받아 기존 난수 키를 유지한다. 학습과 이전 상태가 필요한 구조는 이 고정 그래프에 억지로 넣지 않는다.

## 사용 예

작업 디렉터리: `studies/interaction/`.

```python
from evosim_runtime_v01.storage import Store, TableCodec
from evosim_runtime_v01.records import LineCollectRecords

ordinary = Store(level=9)  # 원래 JSON 구조 유지
archival = Store(TableCodec(LineCollectRecords()), level=3)
archival.write('new_round.json.gz', complete_observed_record)
restored = archival.read('new_round.json.gz')
```

완료 파일은 기본적으로 덮어쓰지 않는다. 동일한 내용을 다시 쓰면 이미 완료됐다는 결과를 반환하고, 내용이 다르면 거부한다. `replace=True`는 체크포인트처럼 의도적으로 교체할 파일만 쓴다. 같은 디렉터리 임시 파일·flush/fsync 후 교체한다. Windows는 기존 대상이 있으면 실패하는 rename, POSIX는 하드링크로 독점 게시한다. 동시 작성자 충돌은 실패로 드러낸다. 모든 네트워크 파일시스템의 장애 내구성까지 보장하지 않는다.

기록은 에피소드·반복 순서, 유전체, 행동열을 공유한다. 알 수 없는 필드·누락 필드·실제로 잘못된 보상·형식 차이를 예외값으로 보존한다. 유효하지 않은 원래 행동 기록은 원문 그대로 저장한다. 따라서 저장 codec은 로그가 과학적으로 맞다는 감사기가 아니다. 복원한 원본에 별도의 감사기를 적용해야 한다.

전체 JSON 해시와 환경 코드 해시를 검사한다. 해시는 손상/불일치 검출용이며 악의적 재작성의 인증 수단은 아니다. 해시 값만 남기고 행동·원본 값 자체를 버리지 않는다. 기존 gzip을 더 강하게 압축하는 기능과 환경별 구조 축소를 구분한다.

## 검증

```powershell
python -B -X utf8 ../../tools/check_q1011.py --output optimization_q1011_recheck_v01
```

새 출력 경로를 사용한다. 완료 소스는 [Q1-011 계획](../../../docs/protocols/Q1-011_실험기반_최적화계획.md)과 기술 측정의 source manifest로 잠근다. 동결 뒤 알고리즘·schema·검사를 바꾸려면 새 버전으로 분리한다.
