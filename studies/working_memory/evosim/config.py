# --- 설정 ---
CONFIG = {
    # 시뮬레이션 기본 설정
    "POPULATION_SIZE": 20,
    "MAX_GENERATIONS": 100,
    "EPISODES_PER_EVAL": 18,       # 3개 클래스 균등 평가 (클래스당 6에피소드)
    "STEPS_PER_EPISODE": 4,        # 다중 스텝 MDP/POMDP (t=0: 단서 제시, t=1..3: 지연/기억 유지 구간)

    # 커리큘럼: 노이즈(noise)뿐만 아니라 센서 마스킹 지연 스텝(mask_steps)을 점진적으로 확대
    "CURRICULUM_STAGES": [
        {"noise": 0.00, "mask_steps": 1, "threshold": 0.78},  # Stage 1: 단기 기억 입문 (마지막 1스텝 마스킹)
        {"noise": 0.08, "mask_steps": 2, "threshold": 0.74},  # Stage 2: 중간 기억 유지 (후반 2스텝 마스킹 + 노이즈)
        {"noise": 0.15, "mask_steps": 3, "threshold": 0.70},  # Stage 3: 심층 작업 기억 (t=0만 단서 제시, 3스텝 연속 마스킹)
    ],
    "STAGE_SUCCESS_THRESHOLD": 0.74,

    # 진화 하이퍼파라미터
    "ELITE_RATIO": 0.2,
    "CROSSOVER_RATE": 0.65,
    "MUTATION_RATE": 0.50,
    "TOURNAMENT_SIZE": 3,
    "COMPLEXITY_PENALTY": 0.005,   # 네트워크 비대화(Bloat) 방지를 위한 간선/모듈 복잡도 페널티

    # 강화학습 (다중 스텝 TD Q-Learning + 보상 조절 출력 가중치 가소성)
    "TD_LEARNING_RATE": 0.18,
    "TD_DISCOUNT": 0.90,
    "TD_WEIGHT": 0.30,             # 신경망 출력과 TD Q-테이블 결합 비중
    "HEBBIAN_LR": 0.015,           # TD 오차(delta) 기반 생애 주기 출력 가중치 미세 조정 학습률
    "LAMARCKIAN_DECAY": 0.50,      # 세대 간 학습 가중치/가치 전이 비율
    "STATE_BINS": 5,
    # Q-테이블 상태 키에 네트워크 내부 기억(t-1 모듈 활성값, 직전 출력 argmax)을 포함할지 여부.
    # False(기본): 두 집단 모두 관측값만 사용 → 수작업 상태 키 기억 채널 제거
    # True: 레거시 동작 (순환 허용 개체의 Q-테이블이 수작업 메모리 채널을 가짐)
    "Q_STATE_INCLUDES_MEMORY": False,
    "EXPERIENCE_BUFFER_SIZE": 120,

    # 구조 및 순환(Recurrent) 메모리 설정
    "INITIAL_MODULES": 4,
    "MAX_MODULES": 9,
    "ALLOW_RECURRENT": True,       # 시간 지연 순환 연결(t-1 -> t) 및 모듈 내부 기억 기본 허용 여부
    # 향후 분리 실험(Ablation)을 위한 세부 스위치 (None이면 개체의 allow_recurrent 설정을 그대로 따름)
    "ALLOW_RECURRENT_EDGES": None, # 그래프 수준 순환 간선(t-1 -> t) 허용 여부 (None | True | False)
    "ALLOW_MODULE_MEMORY": None,   # 모듈 내부 상태 유지(ema/latch) 허용 여부 (None | True | False)
    # 0세대 초기 순환 간선 초기화 모드:
    # - "seeded_positive": 기존 기본 동작 (출력 및 모듈 자기순환에 양수 가중치 시딩; add_module 변이 시에도 양수 시딩 유지)
    # - "zero_mean": 동일한 초기 순환 간선 구조이나 가중치를 평균 0 대칭 구간(-1.0 ~ 1.0)에서 초기화
    # - "none": 0세대 초기 순환 간선을 생성하지 않음 (단, 모듈 내부 기억이나 변이 시 간선 추가와는 별개)
    "INITIAL_RECURRENCE_MODE": "seeded_positive",
    "RECURRENT_MUTATION_PROB": 0.22,

    # 1단계: 동결 검증(Validation) 및 테스트(Test) 평가 분리 설정
    "VALIDATION_EPISODES": 30,          # 후보 개체 선발용 Stage 3 동결 검증 에피소드 수 (3의 배수)
    "TEST_EPISODES": 120,               # 최종 보고용 독립 Stage 3 동결 테스트 에피소드 수 (3의 배수)
    "VALIDATION_SEED": 10001,           # 공통 검증 에피소드 생성 및 행동 선택용 독립 시드
    "TEST_SEED": 20002,                 # 1단계 탐색/검증용 테스트 에피소드 시드
    "CONFIRMATORY_TEST_SEED": 30003,    # 사전 등록된 본 실험(2·3단계 확인적 보고) 전용 미노출 독립 테스트 시드
    "PRE_REGISTERED_SEEDS": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],

    # 보상 체계 (범주형 대칭 보상 + 소프트맥스 마진 Shaping)
    "REWARD_CORRECT": 1.0,
    "REWARD_WRONG": -0.25,
    "MARGIN_SHAPING_SCALE": 0.20,
}
