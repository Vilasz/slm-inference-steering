"""Check whether identical observed prefixes force tied deterministic predictions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from slm_steering.excellence.audit import assert_preserved
from slm_steering.research.protocol import now, read_bank, write_json


def summarize_prefixes(records, predictions, checkpoint):
    selected = predictions[predictions.checkpoint.astype(str) == str(checkpoint)].set_index(['task_id', 'attempt_id'])
    rows = []
    for record in records:
        if record['generated_tokens'] <= checkpoint:
            continue
        p = selected.loc[(record['task_id'], record['attempt_id'])]
        rows.append({'task_id': record['task_id'], 'passed': int(record['passed']),
                     'prefix': tuple(record['token_ids'][:checkpoint]), 'HB': float(p.HB)})
    frame = pd.DataFrame(rows)
    task_rows = []
    for task, group in frame.groupby('task_id', sort=True):
        positive, negative = group[group.passed == 1], group[group.passed == 0]
        mixed = len(positive) > 0 and len(negative) > 0
        unavoidable = sum(a == b for a in positive.prefix for b in negative.prefix)
        for _, same in group.groupby('prefix'):
            if np.ptp(same.HB.to_numpy()) > 1e-9:
                raise ValueError('Identical prompt/prefix has inconsistent deterministic scores; inspect numerical or temporal provenance')
        task_rows.append({'task_id': task, 'checkpoint': checkpoint, 'active_attempts': len(group),
                          'distinct_prefixes': group.prefix.nunique(), 'mixed': mixed,
                          'single_prefix': group.prefix.nunique() == 1,
                          'fraction_correct_wrong_pairs_with_identical_prefix': unavoidable / (len(positive) * len(negative)) if mixed else None})
    tasks = pd.DataFrame(task_rows)
    return {'checkpoint': checkpoint, 'tasks': len(tasks), 'mixed_tasks': int(tasks.mixed.sum()),
            'single_prefix_tasks': int(tasks.single_prefix.sum()),
            'mixed_single_prefix_tasks': int((tasks.mixed & tasks.single_prefix).sum()),
            'mean_fraction_unavoidably_tied_pairs': float(tasks.loc[tasks.mixed, 'fraction_correct_wrong_pairs_with_identical_prefix'].mean()),
            'constant_score_checks_passed': True}, task_rows


def run(root: Path):
    assert_preserved(root)
    output = root / 'reports/excellence/prefixes'
    output.mkdir(exist_ok=True)
    summaries, rows = [], []
    for dataset, phase, path in [('HumanEval+', 'definitive', 'runs/probes/protocol_v0.2/outer_predictions.csv'),
                                 ('MBPP+', 'transfer', 'reports/transfer_v0.2/predictions.csv')]:
        records = read_bank(root / f'runs/candidate_bank/{phase}_v0.2/trajectories.jsonl')
        predictions = pd.read_csv(root / path)
        for checkpoint in (8, 16, 32, 64):
            summary, tasks = summarize_prefixes(records, predictions, checkpoint)
            summaries.append({'dataset': dataset, **summary})
            rows.extend([{'dataset': dataset, **r} for r in tasks])
    pd.DataFrame(rows).to_csv(output / 'per_task.csv', index=False)
    write_json(output / 'results.json', {'created_at': now(), 'status': 'completed_prefix_integrity_diagnostic',
                                        'summaries': summaries,
                                        'scope': 'Identical prompt and token prefix imply tied deterministic scores; partial collisions do not establish a causal explanation of transfer loss',
                                        'primary_source_all_mixed_tasks_have_multiple_prefixes': all(not r['mixed_single_prefix_tasks'] for r in summaries if r['dataset'] == 'HumanEval+' and r['checkpoint'] == 32)})
    assert_preserved(root)
