"""기술 검증의 실제 완료 호출과 미확정 상한. 프로세스마다 독립적이다."""
COUNTS = {'reference_episodes': 0, 'optimized_episodes': 0,
          'completed_step_actions': 0, 'rng_objects_created': 0,
          'unknown_failed_episodes_upper': 0}


def completed(kind, results):
    COUNTS[kind + '_episodes'] += sum(len(records) for _, records in results)
    COUNTS['completed_step_actions'] += sum(len(r['actions']) for _, rows in results for r in rows)
