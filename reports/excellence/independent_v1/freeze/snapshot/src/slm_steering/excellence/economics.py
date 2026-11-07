"""Independent replay, label-aware retrospective oracle, and finite-budget economics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from slm_steering.excellence.audit import assert_preserved
from slm_steering.research.protocol import now, read_bank, write_json
from slm_steering.research.statistics import paired_task_bootstrap


def expected_cost(mean_attempt_cost: float, success_probability: float, budget: int) -> float:
    if not 0 <= success_probability <= 1 or budget < 1 or mean_attempt_cost < 0:
        raise ValueError('Invalid cost, success probability or attempt budget')
    return float(mean_attempt_cost * sum((1 - success_probability) ** j for j in range(budget)))


def simulate(stream, *, checkpoint, threshold=0., scores=None, budget=20, kind='verifier'):
    """Independently implemented accounting, with no original replay function call."""
    if kind not in ('fixed', 'verifier', 'gate', 'oracle') or budget < 1 or checkpoint < 1:
        raise ValueError('Invalid replay definition')
    stream = sorted(stream, key=lambda r: r['attempt_id'])[:budget]
    if len(stream) != budget or len({r['attempt_id'] for r in stream}) != budget:
        raise ValueError('Incomplete or duplicate candidate stream')
    tokens, resolved, started, correct_started, active_correct_started, false_aborts = 0, False, 0, 0, 0, 0
    decisions = []
    for r in stream:
        active = r['generated_tokens'] > checkpoint
        abort = False
        if kind == 'oracle':
            abort = active and not r['passed']
        elif kind == 'gate' and active:
            probability = scores[(r['task_id'], r['attempt_id'])]
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Invalid gate probability')
            abort = probability < threshold
        charged = checkpoint if abort else r['generated_tokens']
        tokens += charged
        started += 1
        correct_started += bool(r['passed'])
        active_correct_started += bool(r['passed']) and active
        false_aborts += bool(r['passed']) and abort
        resolved = resolved or (bool(r['passed']) and not abort)
        decisions.append({'attempt_id': r['attempt_id'], 'aborted': bool(abort), 'tokens': charged})
        if resolved and kind != 'fixed':
            break
    all_correct = [r for r in stream if r['passed'] and r['generated_tokens'] > checkpoint]
    retained = len(all_correct)
    if kind == 'gate':
        retained = sum(scores[(r['task_id'], r['attempt_id'])] >= threshold for r in all_correct)
    return {'task_id': stream[0]['task_id'], 'tokens': tokens, 'resolved': int(resolved), 'started': started,
            'correct_started': correct_started, 'active_correct_started': active_correct_started,
            'false_aborts': false_aborts, 'active_correct_in_budget': len(all_correct),
            'retained_correct_in_budget': int(retained), 'decisions': decisions}


def checkpoint_policy(candidates, checkpoint, tolerance=1e-9):
    """Pick only among the original source-development admissible options at this t."""
    options = [c for c in candidates if c['checkpoint'] == checkpoint]
    if not options:
        raise ValueError('No source-development policy at this checkpoint')
    minimum = min(c['mean_tokens'] for c in options)
    tied = [c for c in options if abs(c['mean_tokens'] - minimum) <= tolerance]
    return min(tied, key=lambda c: (-c['resolution'], c['checkpoint'], c['threshold']))


def paired_ratio_saving(cost, baseline, *, replications=2000, seed=20260921):
    cost, baseline = np.asarray(cost), np.asarray(baseline)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(cost), size=(replications, len(cost)))
    values = 1 - cost[index].sum(axis=1) / baseline[index].sum(axis=1)
    return {'estimate': float(1 - cost.mean() / baseline.mean()), 'ci95': np.quantile(values, [.025, .975]).tolist(),
            'tasks': len(cost), 'unit': 'ratio of task-average costs'}


def run(root: Path):
    assert_preserved(root)
    output = root / 'reports/excellence/economics'
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((root / 'runs/probes/protocol_v0.2/selection_audit.json').read_text(encoding='utf-8'))
    source_policies = {(r['fold'], r['policy']): r for r in selection if 'policy' in r}
    target_metadata = json.loads((root / 'runs/transfer/protocol_v0.2/models.json').read_text(encoding='utf-8'))
    rows, selected_rows = [], []
    for dataset, phase, prediction_path in [('HumanEval+', 'definitive', 'runs/probes/protocol_v0.2/outer_predictions.csv'),
                                            ('MBPP+', 'transfer', 'reports/transfer_v0.2/predictions.csv')]:
        records = read_bank(root / f'runs/candidate_bank/{phase}_v0.2/trajectories.jsonl')
        frame = pd.read_csv(root / prediction_path)
        frame['checkpoint'] = frame.checkpoint.astype(str)
        folds = frame.groupby('task_id').fold.first().to_dict() if 'fold' in frame else {}
        lookups = {(family, t): {(r.task_id, r.attempt_id): float(getattr(r, family))
                                for r in frame[frame.checkpoint == str(t)].itertuples()}
                   for family in ('B', 'HB') for t in (8, 16, 32, 64)}
        by_task = {}
        for r in records:
            by_task.setdefault(r['task_id'], []).append(r)
        for task, stream in sorted(by_task.items()):
            policies = {family: source_policies[(folds[task], family)] if folds else target_metadata['policies'][family]
                        for family in ('B', 'HB')}
            for budget in (1, 5, 10, 20):
                for family, p in policies.items():
                    row = simulate(stream, checkpoint=p['checkpoint'], threshold=p['threshold'], scores=lookups[(family, p['checkpoint'])],
                                   budget=budget, kind='gate')
                    selected_rows.append({**row, 'dataset': dataset, 'budget': budget, 'checkpoint': p['checkpoint'],
                                          'policy': family, 'threshold': p['threshold']})
                for checkpoint in (8, 16, 32, 64):
                    for policy, kind in [('fixed', 'fixed'), ('verifier', 'verifier'), ('oracle', 'oracle'), ('B', 'gate'), ('HB', 'gate')]:
                        p = checkpoint_policy(policies[policy]['candidates'], checkpoint) if kind == 'gate' else None
                        row = simulate(stream, checkpoint=checkpoint, threshold=p['threshold'] if p else 0,
                                       scores=lookups[(policy, checkpoint)] if p else None, budget=budget, kind=kind)
                        rows.append({**row, 'dataset': dataset, 'budget': budget, 'checkpoint': checkpoint,
                                     'policy': policy, 'threshold': p['threshold'] if p else None})
        # Compare the independently replayed original selected gates to archived task outcomes.
        saved_path = 'runs/probes/protocol_v0.2/policy_replay.json' if phase == 'definitive' else 'reports/transfer_v0.2/policy_replay.json'
        saved = json.loads((root / saved_path).read_text(encoding='utf-8'))
        archived = {(r['task_id'], r['budget'], r['policy']): r for r in saved if r['policy'] in ('B', 'HB')}
        for row in [r for r in selected_rows if r['dataset'] == dataset]:
            previous = archived[(row['task_id'], row['budget'], row['policy'])]
            if (row['tokens'], row['resolved'], row['started'], row['false_aborts']) != (previous['completion_tokens'], previous['resolved'], previous['attempts_started'], previous['correct_false_aborts']):
                raise ValueError('Independent replay differs from the archived original policy')
    frame = pd.DataFrame(rows)
    frame.drop(columns='decisions').to_csv(output / 'per_task.csv', index=False)
    pd.DataFrame(selected_rows).drop(columns='decisions').to_csv(output / 'original_selected_policies.csv', index=False)
    summaries = []
    for (dataset, checkpoint, budget), group in frame.groupby(['dataset', 'checkpoint', 'budget']):
        verifier = group[group.policy == 'verifier'].set_index('task_id').sort_index()
        oracle = group[group.policy == 'oracle'].set_index('task_id').reindex(verifier.index)
        if not np.array_equal(verifier.resolved, oracle.resolved) or not (oracle.tokens <= verifier.tokens).all():
            raise ValueError('Oracle must preserve verifier resolution and weakly reduce tokens pathwise')
        for policy, part in group.groupby('policy'):
            part = part.set_index('task_id').reindex(verifier.index)
            denom = part.active_correct_in_budget.sum()
            summaries.append({'dataset': dataset, 'checkpoint': int(checkpoint), 'budget': int(budget), 'policy': policy,
                              'tasks': len(part), 'resolution': float(part.resolved.mean()), 'tokens_per_task': float(part.tokens.mean()),
                              'attempts_started': float(part.started.mean()),
                              'active_correct_retention_all_candidates': float(part.retained_correct_in_budget.sum() / denom) if denom else None,
                              'false_abort_rate_started_active_correct': float(part.false_aborts.sum() / part.active_correct_started.sum()) if part.active_correct_started.sum() else None,
                              'saving_vs_verifier': paired_ratio_saving(part.tokens.to_numpy(), verifier.tokens.to_numpy()),
                              'delta_tokens_vs_verifier': paired_task_bootstrap((part.tokens - verifier.tokens).to_numpy(), replications=2000, seed=20260921),
                              'delta_resolution_vs_verifier': paired_task_bootstrap((part.resolved - verifier.resolved).to_numpy(), replications=2000, seed=20260921),
                              'tokens_minus_oracle': float((part.tokens - oracle.tokens).mean())})
    result = {'created_at': now(), 'status': 'completed_retrospective_oracle', 'summaries': summaries,
              'independent_original_replay_matches': True,
              'oracle_scope': 'Idealized label-aware lower cost for fixed-t policies preserving every correct candidate on the same ordered stream; not a universal lower bound for policies allowed to discard successes',
              'threshold_scope': 'Every t-specific threshold chosen from archived admissible SOURCE-development options only',
              'token_scope': 'Completion tokens; excludes prompt, probe, verification, latency, FLOPs and energy',
              'iid_scope': 'Economic formulas condition on a task; attempts must be i.i.d. across attempts, but within-attempt cost and success may depend'}
    write_json(output / 'results.json', result)
    plot(output, pd.DataFrame(summaries))
    assert_preserved(root)


def plot(output, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {'fixed': '#888888', 'verifier': '#3274A1', 'B': '#E1812C', 'HB': '#3A923A', 'oracle': '#C44E52'}
    for ax, dataset in zip(axes, ('HumanEval+', 'MBPP+')):
        part = summaries[(summaries.dataset == dataset) & (summaries.checkpoint == 8)]
        for policy, group in part.groupby('policy'):
            group = group.sort_values('budget')
            ax.plot(group.tokens_per_task, group.resolution, 'o-', color=colors[policy], label=policy)
        ax.set(xlabel='Tokens de continuação por tarefa', ylabel='Resolução', title=f'{dataset}, checkpoint 8')
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'oracle_frontier.{ext}', dpi=180)
    plt.close(fig)
