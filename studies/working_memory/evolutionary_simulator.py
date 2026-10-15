"""
하위 호환용 실행 스크립트.

실제 구현은 `evosim/` 패키지로 분리되었습니다:
    evosim/config.py        - 하이퍼파라미터(CONFIG)
    evosim/innovation.py    - InnovationTracker
    evosim/logger.py        - Logger
    evosim/environment.py   - Environment (다중 스텝 POMDP)
    evosim/module.py        - Module (노드 연산자)
    evosim/entity/          - Entity (network / learning / mutation Mixin 구성)
    evosim/simulator.py     - EvolutionarySimulator
    evosim/experiment.py    - run_working_memory_research

실행: `python evolutionary_simulator.py` 또는 `python -m evosim`
"""

from evosim import *  # noqa: F401,F403  (기존 import 경로 호환)
from evosim import run_working_memory_research

if __name__ == "__main__":
    run_working_memory_research()