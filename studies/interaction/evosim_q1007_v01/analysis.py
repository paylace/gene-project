"""사전 이진 대응 검정과 고정 두 대비 Holm 판정."""
import math
import numpy as np


def paired_p(wins, losses, two_sided=False):
    n = wins + losses
    if not n:
        return 1.
    start = max(wins, losses) if two_sided else wins
    tail = sum(math.comb(n, k) for k in range(start, n + 1)) / 2 ** n
    return min(1., tail * (2 if two_sided else 1))


def holm(values):
    ordered = sorted(values, key=lambda key: (values[key], key))
    result, previous = {}, 0.
    for index, key in enumerate(ordered):
        previous = max(previous, min(1., (len(values) - index) * values[key]))
        result[key] = previous
    return result


def decision(rates, *, active=None, sample_size=80):
    names = ('PT-B', 'P-T')
    active = set(names if active is None else active)
    details, pvalues = {}, {}
    for name in names:
        if name not in active:
            pvalues[name] = 1.
            continue
        left, right = name.split('-')
        if len(rates[left]) != sample_size or len(rates[right]) != sample_size:
            raise ValueError('고정 시드 수 불일치')
        a, b = (np.asarray(rates[k], dtype=float) for k in (left, right))
        if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any((a < 0) | (a > 1)) or np.any((b < 0) | (b > 1)):
            raise ValueError('완료율 범위 오류')
        d = (a >= .8 - 1e-12).astype(int) - (b >= .8 - 1e-12).astype(int)
        wins, losses = int((d > 0).sum()), int((d < 0).sum())
        pvalues[name] = paired_p(wins, losses, name == 'P-T')
        rng = np.random.default_rng(72007)
        boot = d[rng.integers(0, sample_size, size=(10000, sample_size))].mean(axis=1)
        details[name] = {'wins': wins, 'losses': losses, 'ties': sample_size - wins - losses,
                         'discovery_fraction_difference': float(d.mean()), 'raw_p': pvalues[name],
                         'ci95_descriptive': np.quantile(boot, [.025, .975]).tolist(),
                         'practical': (abs(float(d.mean())) if name == 'P-T' else float(d.mean())) >= .1 - 1e-12}
    adjusted = holm(pvalues)
    eligible = []
    for name, row in details.items():
        row['holm_p'] = adjusted[name]
        row['passed'] = row['practical'] and adjusted[name] <= .05
        if row['passed']:
            eligible.append(name)
    return {'contrasts': details, 'eligible': sorted(eligible), 'family': list(names), 'automatic_adoption': False}
