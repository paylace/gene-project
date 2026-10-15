"""Validate all saved trials and create readable results without rerunning evolution."""
import json
from pathlib import Path
import statistics

from run_research import CONDITIONS, OUT, SEEDS, atomic_json, bootstrap_ci, describe

LABELS = {
    'baseline': '기본: Q 결합 + 가소성 + 두 기억 구조',
    'no_q_action': 'Q 행동 결합 끔',
    'no_plasticity': '가중치 가소성 끔',
    'neither_learning_path': 'Q 행동 결합·가소성 모두 끔',
    'edges_only': '그래프 순환만 허용',
    'modules_only': '모듈 내부 기억만 허용',
    'no_memory': '두 기억 구조 모두 차단',
    'init_zero_mean': '초기 순환 가중치 평균 0',
    'init_no_edges': '초기 순환 간선 없음, 내부 기억 허용',
    'edges_only_init_none': '초기 순환·내부 기억 없음, 순환 변이는 허용',
}


def load():
    rows = {}
    for condition in CONDITIONS:
        rows[condition] = []
        for seed in SEEDS:
            row = json.loads((OUT / condition / f'{seed}.json').read_text(encoding='utf-8'))
            assert row['seed'] == seed and row['condition'] == condition
            assert len(row['generations']) == 100
            flags = row['summary']['active_memory_flags']
            assert flags['allow_recurrent_edges'] == CONDITIONS[condition]['edges']
            assert flags['allow_module_memory'] == CONDITIONS[condition]['memory']
            for phase in ['historical', 'final', 'gen0_selected_pre', 'gen0_same_post', 'gen0_selected_post']:
                metrics = row[phase]
                episodes = metrics['episodes']
                assert len(episodes) == 120
                assert all(sum(e['target'] == c for e in episodes) == 40 for c in range(3))
                masked = [s['correct'] for e in episodes for s in e['steps'] if s['masked']]
                full = [all(s['correct'] for s in e['steps'] if s['masked']) for e in episodes]
                assert abs(statistics.mean(masked) - metrics['memory_accuracy']) < 1e-12
                assert abs(statistics.mean(full) - metrics['episode_full_memory_accuracy']) < 1e-12
                for t in range(4):
                    value = statistics.mean(e['steps'][t]['correct'] for e in episodes)
                    assert abs(value - metrics['step_accuracies'][str(t)]) < 1e-12
            rows[condition].append(row)
    return rows


def paired(differences):
    return dict(**describe(differences), paired_bootstrap_95=bootstrap_ci(differences), values=differences)


def main():
    rows = load()
    base = rows['baseline']
    extra = dict(
        dependency={key: paired([r['dependency'][key]['memory_accuracy'] - r['historical']['memory_accuracy']
                                 for r in base]) for key in base[0]['dependency']},
        adaptation_plasticity={str(q): paired([
            r['adaptation'][f'hebb=0.015,q={q}']['memory_accuracy']
            - r['adaptation'][f'hebb=0.0,q={q}']['memory_accuracy'] for r in base]) for q in (0.0, 0.3)},
        lifetime_paired_gen0=paired([r['gen0_same_post']['memory_accuracy']
                                    - r['gen0_selected_pre']['memory_accuracy'] for r in base]),
        lifetime_population_gen0=paired([r['gen0_lifetime_mean_delta'] for r in base]),
        final_minus_historical=paired([r['final']['memory_accuracy'] - r['historical']['memory_accuracy']
                                      for r in base]),
        adaptation_note='Historical pre-lifetime candidate was selected through its original post-learning validation; this diagnostic is conditional on that selection.',
    )
    atomic_json(OUT / 'diagnostics.json', extra)
    def fmt(vals):
        return f'{100*statistics.mean(vals):.2f}% ± {100*statistics.stdev(vals):.2f}%'
    lines = ['# 실행 결과', '',
             '10개 조건 × 시드 101~110 × 100세대. 집단 크기 20, 매 세대 생애 학습 18에피소드.', '',
             '선발: Stage 3 검증 30에피소드(시드 10001), 매 세대 전체 집단에서 적합도·정확도·기억 정확도 순.',
             '주 평가: 별도 Stage 3 테스트 120에피소드(시드 30003), 가중치·Q값 동결.',
             '보조 일반화 평가: 시드 30004~30008의 총 600에피소드. 후보를 다시 고르는 데 사용하지 않음.', '',
             '각 ±는 10개 진화 시드 간 표본 표준편차. 같은 테스트셋에서의 시드 변동이며 환경 전체의 오차 한계가 아님.', '',
             '| 조건 | 실행 중 선발 후보 기억 정확도 | 마지막 세대 후보 | 에피소드 기억 완주율 | 보조 테스트 기억 정확도 |',
             '|---|---:|---:|---:|---:|']
    for condition, group in rows.items():
        historical = [r['historical']['memory_accuracy'] for r in group]
        final = [r['final']['memory_accuracy'] for r in group]
        full = [r['historical']['episode_full_memory_accuracy'] for r in group]
        secondary = [statistics.mean(v['memory_accuracy'] for v in r['secondary_test_sets'].values()) for r in group]
        lines.append(f'| {LABELS[condition]} | {fmt(historical)} | {fmt(final)} | {fmt(full)} | {fmt(secondary)} |')
    lines += ['', '## 같은 기본 개체의 기능을 평가 중에만 끈 결과', '',
              '| 평가 변경 | 기억 정확도 변화, %p | 시드 짝 부트스트랩 95% 구간, %p |',
              '|---|---:|---:|']
    for key, value in extra['dependency'].items():
        lo, hi = value['paired_bootstrap_95']
        label = {'q_off': 'Q 행동 결합 제거', 'edges_off': '그래프 순환 제거',
                 'modules_off': '모듈 기억 OFF(연산 함수도 변경됨)', 'all_memory_off': '두 기억 구조 OFF(연산 함수도 변경됨)',
                 'pre_lifetime': '이번 생애 학습 전 상태'}[key]
        lines.append(f"| {label} | {100*value['mean']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ['', '## 해석 범위', '',
              '- 실행 중 선발 후보는 최종 세대 개체와 다를 수 있다. 선택된 세대·단계는 각 시드 JSON에 보관했다.',
              '- 재진화 조건 차이는 진화·학습·상속·커리큘럼·선발 절차 전체의 차이이다.',
              '- 기능 제거는 해당 개체의 의존성, 재진화는 기능이 없는 조건의 적응 가능성을 측정한다.',
              '- 내부 기억 OFF에서는 ema/latch가 tanh로 계산된다. 내부 기억 비교는 시간 상태 제거 외에도 연산자의 순간 전달 함수를 바꾼다.',
              '- 초기 간선 없음은 이후 변이의 양수 자기순환 시딩까지 없앤 조건이 아니다.',
              '- 초기화 조건들은 같은 시드라도 이후 난수 대응과 초기 순방향 가중치가 달라질 수 있다.',
              '- 동일한 30개 검증 에피소드로 많은 후보를 비교한다. 검증 과적합 가능성이 있으므로 독립 테스트를 함께 보관했다.',
              '- 10개 시드의 부트스트랩 구간은 기술적 요약이며 다중 비교 보정이나 모집단 전체에 대한 보증이 아니다.',
              '- 보조 테스트와 학습 중의 기록을 보고 조건을 재선발하지 않았다.', '',
              '재현: `python run_research.py --workers 3`, 결과 점검: `python analyze_research.py`.',
              'protocol.json은 실행 전 고정한 조건과 소스 SHA-256을 담고, 각 조건 폴더는 에피소드별 응답·선택 정책·훈련 이력을 담는다.', '']
    followup = OUT.parent / 'adaptation_followup'
    if (followup / 'summary.json').exists():
        follow_summary = json.loads((followup / 'summary.json').read_text(encoding='utf-8'))
        targets = [e[0]['correct_action'] for e in json.loads((followup / 'datasets.json').read_text(encoding='utf-8'))['test']]
        for seed in SEEDS:
            trial = json.loads((followup / f'{seed}.json').read_text(encoding='utf-8'))
            scores = list(trial['before'].values()) + [m for branch in trial['branches'].values() for m in branch.values()]
            for score in scores:
                assert len(score['episode_actions']) == 1200
                masked = [a == target for actions, target in zip(score['episode_actions'], targets) for a in actions[1:]]
                assert abs(statistics.mean(masked) - score['memory_accuracy']) < 1e-12
        lines += ['## 탐색적 후속 실험: TD 부트스트랩과 장기 적응', '',
                  '기본 조건에서 선택된 동일 개체의 생애 학습 전 상태에서 분기했다. 같은 180에피소드 훈련 묶음의 18/180 prefix를 사용했다.',
                  '새 테스트는 시드 30009의 1,200에피소드이다. 아래 값은 Q 행동 결합을 끈 동결 평가에서 학습 전 대비 변화이며 가중치 변화 경로를 본다.',
                  '학습 중 Q 행동 결합은 모든 분기에서 0.3이었다. 이 실험은 본 실험을 본 후 설계한 탐색적 비교이며 후보 재선발은 하지 않았다.', '',
                  '| 적응 에피소드 | TD 할인율 | 가소성 ON 평균 변화, %p | 시드 짝 부트스트랩 95% 구간, %p |',
                  '|---|---|---:|---:|']
        for n in (18, 180):
            for gamma in (0.0, 0.9):
                stat = follow_summary['branches'][f'n={n},gamma={gamma},hebb=0.015']
                lo, hi = stat['paired_bootstrap_95']
                lines.append(f"| {n} | {gamma} | {100*stat['mean']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
        lines += ['', '18에피소드에서 학습 이득은 작고 불확실했다. 180에피소드에서는 두 할인율 모두 평균 성능이 하락했다.',
                  '따라서 부트스트랩을 제거하기만 하면 현재 가소성 문제가 해결된다는 근거는 얻지 못했다.',
                  '가소성 OFF·Q 행동 결합 OFF 조건은 모든 시드에서 학습 전과 에피소드별 응답이 완전히 동일했다.', '']
    delay_folder = OUT.parent / 'delay_probe'
    if (delay_folder / 'summary.json').exists():
        delay_stats = json.loads((delay_folder / 'summary.json').read_text(encoding='utf-8'))
        targets = [e[0]['correct_action'] for e in json.loads((delay_folder / 'datasets.json').read_text(encoding='utf-8'))]
        lines += ['## 학습 지연을 넘어서 유지되는가?', '',
                  '새 시드 50001에서 생성한 33스텝 궤적의 prefix를 사용해, 각 지연 조건에서 단서와 앞부분 배경 잡음을 동일하게 유지했다.',
                  '후보를 다시 선택하거나 재학습하지 않았다. 아래는 마스킹 전체 평균이 아닌 마지막 시점의 정답률이다.', '',
                  '| 조건 | 1스텝 | 3스텝 | 5스텝 | 8스텝 | 16스텝 | 32스텝 |', '|---|---:|---:|---:|---:|---:|---:|']
        for name, times in delay_stats.items():
            lines.append('| '+LABELS[name]+' | '+' | '.join(f"{100*times[str(d)]['mean']:.2f}%" for d in [1,3,5,8,16,32])+' |')
            for seed in SEEDS:
                trial = json.loads((delay_folder / f'{name}_{seed}.json').read_text(encoding='utf-8'))
                for delay, score in trial['measurements'].items():
                    actions = score['episode_actions']
                    assert all(len(a) == int(delay)+1 for a in actions)
                    expected = statistics.mean(a[-1] == t for a,t in zip(actions,targets))
                    assert abs(expected-score['last_masked_accuracy']) < 1e-12
        lines += ['', '3스텝 과제의 높은 성능이 긴 지연의 안정적인 기억을 뜻하지는 않는다.', '']
    temporal = OUT.parent / 'temporal_state_probe'
    if (temporal / 'summary.json').exists():
        temporal_stats = json.loads((temporal / 'summary.json').read_text(encoding='utf-8'))
        targets = [e[0]['correct_action'] for e in json.loads((temporal / 'datasets.json').read_text(encoding='utf-8'))]
        for seed in SEEDS:
            trial = json.loads((temporal / f'{seed}.json').read_text(encoding='utf-8'))['measurements']
            for name, score in trial.items():
                expected = statistics.mean(a == t for actions,t in zip(score['episode_actions'],targets) for a in actions[1:])
                assert abs(expected-score['memory_accuracy']) < 1e-12
            for name in ['module_state_zero', 'edges_off', 'all_temporal_state_zero']:
                assert all(a[0] == b[0] for a,b in zip(trial['baseline']['episode_actions'],trial[name]['episode_actions']))
        lines += ['## 모듈의 이전 상태만 제거한 정밀 검사', '',
                  '새 시드 50003, 1,200에피소드. ema/latch 식은 유지하고 prev_self_state 입력만 0으로 만든다.',
                  '이 검사는 기존 기억 OFF 스위치가 tanh로 함수를 바꾸는 효과와 시간 상태 제거 효과를 구분한다.', '',
                  '| 조작 | 기억 정확도 | 기본 대비 변화, %p | 단서 정확도 |', '|---|---:|---:|---:|']
        for mode, label in [('baseline','기본'),('module_state_zero','모듈 이전 상태만 0'),
                            ('legacy_module_off','기존 모듈 기억 OFF(함수 변경)'),('edges_off','그래프 순환 제거'),
                            ('all_temporal_state_zero','모든 시간 상태 제거(함수 유지)')]:
            score = temporal_stats[mode]
            lines.append(f"| {label} | {100*score['mean']:.2f}% | {100*score['memory_change']:+.2f} | {100*score['cue_mean']:.2f}% |")
        lines += ['', '순수 상태 제거 조작은 10개 개체 × 1,200에피소드에서 단서 시점 행동을 하나도 바꾸지 않았다.',
                  '기존 모듈 OFF의 성능 하락을 전부 내부 기억의 기여도로 해석해서는 안 된다. 기본 개체는 주로 그래프 순환에 의존했다.', '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Validated 100 trials, episode metrics and active flags; wrote RESULTS.md and diagnostics.json.')


if __name__ == '__main__':
    main()
