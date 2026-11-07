"""Development-only repeated full selection/refits; no external labels are read."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from slm_steering.final_extension.common import REPORT, RUN, atomic_json, read_json, digest, canonical
from slm_steering.final_extension.probes import FAMILIES, fit_family, predict, grouped_partition
from slm_steering.research.statistics import task_auc


def run(data, name, repetitions=8):
    fit_mask, audit_mask = grouped_partition(data['groups'], namespace='procedure-audit')
    audit_indices = np.flatnonzero(audit_mask)
    pool = sorted(set(data['groups'][fit_mask]))
    directory = RUN / name / 'training_sensitivity'
    directory.mkdir(parents=True, exist_ok=True)
    rows, selections = [], []
    with threadpool_limits(limits=1):
        for repetition in range(repetitions):
            path = directory / f'repeat_{repetition}.json'
            if path.exists():
                saved = read_json(path)
                if saved['source_contract'] != data['contract']:
                    raise ValueError('Training sensitivity source changed')
                rows.extend(saved['rows'])
                selections.extend(saved['selections'])
                continue
            rng = np.random.default_rng(20261104 + repetition)
            # Task subsampling avoids duplicate originals leaking between folds;
            # this is a perturbation experiment, not a cluster bootstrap CI.
            selected = set(rng.choice(pool, size=int(.8 * len(pool)), replace=False))
            indices = np.flatnonzero(np.isin(data['groups'], sorted(selected)))
            subset = {k: data[k][indices] for k in ('y', 'groups', 'hidden', 'observable')}
            local_rows, choices = [], []
            started = time.perf_counter()
            for family in FAMILIES:
                fitted = fit_family(subset, family)
                probabilities = predict(fitted, data, audit_indices)
                aucs = task_auc(data['y'][audit_indices], probabilities, data['groups'][audit_indices])
                local_rows.append({'repetition': repetition, 'family': family,
                                   'wt_auc': float(np.mean(list(aucs.values()))) if aucs else None,
                                   'task_aucs': aucs})
                choices.append({'repetition': repetition, 'family': family,
                                'parameter': fitted['audit']['chosen']['parameter'],
                                'epochs': fitted['audit']['final_epochs']})
            saved = {'source_contract': data['contract'], 'rows': local_rows, 'selections': choices,
                     'fit_tasks': sorted(selected), 'audit_tasks': sorted(set(data['groups'][audit_indices])),
                     'seconds': time.perf_counter() - started}
            atomic_json(path, saved)
            rows.extend(local_rows)
            selections.extend(choices)
            print(f'{name} full-procedure perturbation {repetition+1}/{repetitions} '
                  f'({saved["seconds"]:.1f}s)', flush=True)
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != 'task_aucs'} for r in rows])
    wide = frame.pivot(index='repetition', columns='family', values='wt_auc')
    for family in ('linear', 'MLP'):
        wide[f'delta_{family}'] = wide[f'HB_{family}'] - wide[f'B_{family}']
    wide.to_csv(REPORT / f'{name}_training_sensitivity.csv')
    result = {'model': name, 'repetitions': repetitions, 'source_contract': data['contract'],
              'audit_task_ids': sorted(set(data['groups'][audit_indices])),
              'training_pool_tasks': len(pool), 'tasks_per_perturbation': int(.8 * len(pool)),
              'training_sensitivity_sd': {f: float(wide[f'delta_{f}'].std(ddof=1)) for f in ('linear', 'MLP')},
              'selection_frequencies': selections,
              'scope': 'Repeated task subsampling and complete source selection/refit/calibration, scored on fixed disjoint source tasks.',
              'limits': 'Eight overlapping subsamples and a small fixed audit cohort do not yield a full-procedure confidence interval or an unbiased training variance estimate.',
              'no_external_labels': True, 'output_sha256': digest(canonical(rows))}
    atomic_json(REPORT / f'{name}_training_sensitivity.json', result)
    return result
