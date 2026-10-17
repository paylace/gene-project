"""이동·수집의 고정 순방향 집단 평가. 유전 연산과 선발은 호출자가 제공한다."""
import random
import numpy as np
from .accounting import COUNTS, completed
from evosim_interaction_v02.rollout import derive_seed


class FixedTanhBackend:
    """동일한4노드 그래프. 합산 순서 보존, 행렬 곱/학습/후보 간 상태 공유 없음."""
    @staticmethod
    def supports(genome):
        modules = tuple('m' + str(i) for i in range(4))
        inputs = tuple(f'in_{i}' for i in range(4))
        outputs = tuple(f'out_{i}' for i in range(3))
        expected = {m: set(inputs) for m in modules}
        expected.update({o: set(modules) for o in outputs})
        return (not genome.rec and set(genome.modules) == set(modules)
                and {k: set(v) for k, v in genome.ff.items()} == expected
                and all(m.op == 'tanh' for m in genome.modules.values()))

    def __init__(self, genomes):
        if not genomes or not all(self.supports(g) for g in genomes):
            raise ValueError('고정4모듈 tanh 그래프만 지원')
        self.weights = np.array([[[g.ff[f'm{m}'][f'in_{i}'] for i in range(4)]
                                 for m in range(4)] for g in genomes])
        self.biases = np.array([[g.modules[f'm{m}'].bias for m in range(4)] for g in genomes])
        self.readout = np.array([[[g.ff[f'out_{o}'][f'm{m}'] for m in range(4)]
                                 for o in range(3)] for g in genomes])

    def forward(self, obs):
        # obs: 후보×에피소드×4. 기존 BatchNetwork의 += 순서를 그대로 따른다.
        hidden = []
        for m in range(4):
            weighted = np.zeros(obs.shape[:2])
            for i in range(4):
                weighted += obs[:, :, i] * self.weights[:, m, i, None]
            hidden.append(np.tanh(np.clip(weighted + self.biases[:, m, None], -10., 10.)))
        outputs = []
        for o in range(3):
            weighted = np.zeros(obs.shape[:2])
            for m in range(4):
                weighted += hidden[m] * self.readout[:, o, m, None]
            outputs.append(weighted)
        result = np.stack(outputs, axis=-1)
        if not np.isfinite(result).all():
            raise FloatingPointError('유한하지 않은 출력')
        return result


class LazyRngs:
    def __init__(self, rows, purposes, seed):
        self.rows, self.purposes, self.seed = rows, purposes, seed
        self.values = {}

    def __getitem__(self, index):
        index = int(index)
        if index not in self.values:
            candidate, row = divmod(index, len(self.rows))
            episode, rep = self.rows[row]
            self.values[index] = random.Random(derive_seed(self.purposes[candidate], self.seed, episode.episode_id, rep))
            COUNTS['rng_objects_created'] += 1
        return self.values[index]


def select(logits, active, rngs):
    exps = np.exp(np.clip(logits - logits.max(axis=-1, keepdims=True), -10., 10.))
    probs = exps / exps.sum(axis=-1, keepdims=True)
    maxima = probs.max(axis=-1, keepdims=True)
    close = np.abs(probs - maxima) <= np.maximum(1e-9, 1e-9 * np.maximum(np.abs(probs), np.abs(maxima)))
    tied = active & (close.sum(axis=-1) > 1)
    actions = np.argmax(probs, axis=-1)
    flat = actions.reshape(-1)
    choices = close.reshape(-1, 3)
    for row in np.flatnonzero(tied):
        flat[row] = rngs[row].choice(np.flatnonzero(choices[row]).tolist())
    return actions, tied.sum(axis=1)


def evaluate_population(genomes, episodes, seed, purposes, *, settings, metrics,
                        repeats=1, training=False, reference=None, backend='auto'):
    """후보 순서로 (지표, 전체 기록)을 반환한다. 미지원 구조는 명시 경로로 보낸다."""
    genomes, episodes, purposes = tuple(genomes), tuple(episodes), tuple(purposes)
    if (not genomes or not episodes or len(purposes) != len(genomes)
            or type(repeats) is not int or repeats < 1 or backend not in ('auto', 'fixed', 'reference')):
        raise ValueError('잘못된 평가 요청')
    for genome in genomes:
        genome.validate(settings)
    supported = all(FixedTanhBackend.supports(g) for g in genomes)
    if backend == 'reference' or (not supported and backend == 'auto'):
        if reference is None:
            raise ValueError('미지원 구조에는 기준 평가 함수가 필요')
        return [reference(g, episodes, seed, p, repeats=repeats, training=training)
                for g, p in zip(genomes, purposes)]
    if not supported:
        raise ValueError('고정 백엔드에 맞지 않는 구조')
    before = [g.digest for g in genomes]
    rows = [(ep, rep) for rep in range(repeats) for ep in episodes]
    count, size = len(genomes), len(rows)
    maximum_calls = count * size
    try:
        shape = (count, size)
        targets = np.array([ep.targets for ep, _ in rows])
        position = np.broadcast_to([ep.start for ep, _ in rows], shape).copy()
        collected = np.zeros(shape, dtype=int)
        steps, walls, failed = (np.zeros(shape, dtype=int) for _ in range(3))
        minimum = abs(position - targets[:, 0])
        actions_log = np.full((*shape, 24), -1, dtype=np.int8)
        rewards_log = np.zeros((*shape, 24), dtype=np.int16)
        progress_log = np.zeros((*shape, 24), dtype=np.int8) if training else None
        policy, rngs = FixedTanhBackend(genomes), LazyRngs(rows, purposes, seed)
        ties = np.zeros(count, dtype=int)
        for step in range(24):
            active = collected < 3
            if not active.any():
                break
            goal = targets[np.arange(size)[None, :], np.minimum(collected, 2)]
            obs = np.stack((position / 6, goal / 6, (3 - collected) / 3,
                            np.full(shape, (24 - step) / 24)), axis=-1)
            actions, step_ties = select(policy.forward(obs), active, rngs)
            ties += step_ties
            did_collect = active & (actions == 2) & (position == goal)
            bad_collect = active & (actions == 2) & (position != goal)
            proposed = np.clip(position + np.where(actions == 0, -1, np.where(actions == 1, 1, 0)), 0, 6)
            hit_wall = active & (actions != 2) & (proposed == position)
            if training:
                distance = abs(proposed - goal)
                progress_log[:, :, step] = np.where(active & (actions != 2), np.maximum(0, minimum - distance), 0)
                minimum = np.where(active, np.minimum(minimum, distance), minimum)
            position = np.where(active, proposed, position)
            collected += did_collect
            if training:
                next_goal = targets[np.arange(size)[None, :], np.minimum(collected, 2)]
                minimum = np.where(did_collect & (collected < 3), abs(position - next_goal), minimum)
            steps += active
            walls += hit_wall
            failed += bad_collect
            actions_log[:, :, step] = np.where(active, actions, -1)
            rewards_log[:, :, step] = np.where(active, 100 * did_collect.astype(int) - 1, 0)
        ascii_actions = (actions_log + 48).astype(np.uint8).tobytes()
        result = []
        for candidate in range(count):
            records = []
            for row, (episode, rep) in enumerate(rows):
                length = int(steps[candidate, row])
                offset = (candidate * size + row) * 24
                record = dict(episode_id=episode.episode_id, repeat=rep,
                    actions=ascii_actions[offset:offset + length].decode('ascii'),
                    rewards_cents=rewards_log[candidate, row, :length].tolist(),
                    collected=int(collected[candidate, row]), success=bool(collected[candidate, row] == 3),
                    walls=int(walls[candidate, row]), failed_collect=int(failed[candidate, row]),
                    optimal_steps=episode.optimal_steps, final_position=int(position[candidate, row]))
                if training:
                    record['progress_units'] = progress_log[candidate, row, :length].tolist()
                records.append(record)
            measured = metrics(records, training)
            measured['ties'] = int(ties[candidate])
            result.append((measured, records))
        if before != [g.digest for g in genomes]:
            raise AssertionError('평가가 유전체를 변경함')
        completed('optimized', result)
        return result
    except Exception:
        COUNTS['unknown_failed_episodes_upper'] += maximum_calls
        raise
