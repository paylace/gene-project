"""이동·수집 기록의 벡터 복원. 실제 관측과의 차이는 저장 계층이 보존한다."""
import copy
import hashlib
from pathlib import Path
import numpy as np
from evosim_interaction_v02 import environment

BASE_FIELDS = ('actions', 'rewards_cents', 'collected', 'success', 'walls',
               'failed_collect', 'optimal_steps', 'final_position')


class LineCollectRecords:
    schema = 'linecollect-actions-1'

    @property
    def dependencies(self):
        return {'environment_sha256': hashlib.sha256(Path(environment.__file__).read_bytes()).hexdigest()}

    def restore(self, episode_ids, actions, training):
        if not episode_ids or len(episode_ids) != len(actions) or type(training) is not bool:
            raise ValueError('복원 행 수가 맞지 않음')
        specs = []
        for name in episode_ids:
            start, goals = name.split(':')
            spec = environment.EpisodeSpec(int(start), tuple(map(int, goals.split(','))))
            if spec.episode_id != name:
                raise ValueError('표준 에피소드 ID가 아님')
            specs.append(spec)
        lengths = np.array([len(a) for a in actions])
        if any(not isinstance(a, str) or not 1 <= len(a) <= 24 or set(a) - set('012') for a in actions):
            raise ValueError('잘못된 행동열')
        values = np.frombuffer(b''.join(a.encode('ascii').ljust(24, b'/') for a in actions), dtype=np.uint8).reshape(-1, 24).astype(np.int16) - 48
        position = np.array([s.start for s in specs])
        targets = np.array([s.targets for s in specs])
        count = np.zeros(len(actions), dtype=int)
        walls, failed = count.copy(), count.copy()
        minimum = abs(position - targets[:, 0])
        rewards = np.zeros_like(values)
        progress = np.zeros_like(values)
        for step in range(24):
            active = step < lengths
            if np.any(active & (count == 3)):
                raise ValueError('종료 후 행동')
            goal = targets[np.arange(len(actions)), np.minimum(count, 2)]
            action = values[:, step]
            collected = active & (action == 2) & (position == goal)
            proposed = np.clip(position + np.where(action == 0, -1, np.where(action == 1, 1, 0)), 0, 6)
            distance = abs(proposed - goal)
            progress[:, step] = np.where(active & (action != 2), np.maximum(0, minimum - distance), 0)
            minimum = np.where(active, np.minimum(minimum, distance), minimum)
            rewards[:, step] = np.where(active, np.where(collected, 99, -1), 0)
            walls += active & (action != 2) & (proposed == position)
            failed += active & (action == 2) & ~collected
            position = np.where(active, proposed, position)
            count += collected
            next_goal = targets[np.arange(len(actions)), np.minimum(count, 2)]
            minimum = np.where(collected & (count < 3), abs(position - next_goal), minimum)
            if np.any(active & (step == lengths - 1) & (count < 3) & (step != 23)):
                raise ValueError('실패가 제한 시간 전에 끝남')
        result = []
        for i, (name, action, spec) in enumerate(zip(episode_ids, actions, specs)):
            row = dict(actions=action, rewards_cents=rewards[i, :len(action)].tolist(),
                       collected=int(count[i]), success=bool(count[i] == 3), walls=int(walls[i]),
                       failed_collect=int(failed[i]), optimal_steps=spec.optimal_steps,
                       final_position=int(position[i]))
            if training:
                row['progress_units'] = progress[i, :len(action)].tolist()
            result.append(row)
        return result

    def restore_unique(self, order, actions, training):
        # 같은 파일의 같은 증거를 중복 재생하지 않는다. 원본 값의 예외는 별도 적용한다.
        lookup, unique, refs = {}, [], []
        for (name, _), action in zip(order, actions):
            key = (name, action)
            if key not in lookup:
                lookup[key] = len(unique)
                unique.append(key)
            refs.append(lookup[key])
        rows = self.restore([u[0] for u in unique], [u[1] for u in unique], training)
        return [copy.deepcopy(rows[i]) for i in refs]
