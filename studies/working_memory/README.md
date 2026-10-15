# evosim 연구 실행본

전달받은 1단계 v2 패치를 기준으로 한 독립 실행 디렉터리입니다. 원본 전달 폴더는 수정하지 않습니다.

현재 추가된 변경은 명시적 RNG를 사용하는 생애 학습에서 경험 회상까지 동일 RNG로 격리하는 수정과, 실험 실행·검증 도구입니다. 기존 `rng=None` 훈련 경로는 유지합니다.

```powershell
python -m unittest discover -s tests -v
python run_research.py --workers 3
python analyze_research.py
```

후속 검사는 진화 개체를 다시 선발하지 않고 저장된 정책을 사용합니다. 기본 실험이 끝난 뒤 아래 순서로 실행하면 새 데이터셋의 적응·지연·시간 상태 제거 검사까지 재현할 수 있습니다.

```powershell
python run_adaptation_followup.py
python probe_memory_delay.py
python probe_temporal_state.py
python analyze_research.py
```

후속 실험은 본 실험을 확인한 뒤 설계한 탐색적 실험이며, 각각 별도 규약·새 테스트 시드·에피소드별 응답을 저장합니다. `probe_temporal_state.py`는 기존 OFF 스위치의 tanh 치환을 사용하지 않고, 원래 모듈 식을 유지한 채 이전 상태 입력만 0으로 만드는 개입도 비교합니다.

Windows의 python 명령이 실행되지 않는 환경에서는 이번 실행에 사용한 다음 인터프리터로 대체합니다.

```powershell
& 'C:\Users\thoma\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -X utf8 run_research.py --workers 3
```

`research_results/protocol.json`은 조건·시드·후보 선발·평가 절차 및 소스 해시를 실행 전에 저장합니다. 결과가 있는 조건/시드는 건너뛰므로 중단 후 같은 명령으로 재개할 수 있습니다. 소스나 규약을 바꾼 경우 기존 결과와 섞지 않도록 실행을 거부합니다.

`research_results/RESULTS.md`, `summary.json`, `summary.csv`, `diagnostics.json`은 100회 실행 완료 후 생성됩니다. 개별 시드 JSON은 학습곡선, 역사적 후보·마지막 세대 후보의 성능, 에피소드 응답을 저장합니다. `_policies.json`은 선택 개체의 연결·모듈·Q값을 보존합니다.

과거에 남아 있던 모듈 ID의 계통 간 충돌 및 Q와 가중치의 상속 방식은 이번 메커니즘 비교에서 그대로 유지합니다. 그러므로 결과는 현재 알고리즘에 대한 비교이며 NEAT 정합성이나 순수 유전적 원인만의 입증이 아닙니다. 이 문제를 수정하는 후속 버전은 별도 기준선으로 다뤄야 합니다.
