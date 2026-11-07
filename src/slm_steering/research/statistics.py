from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def task_weights(groups) -> np.ndarray:
    groups = np.asarray(groups)
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return len(groups) / (len(unique) * counts[inverse])


def task_auc(y, scores, groups) -> dict[str, float]:
    y, scores, groups = np.asarray(y), np.asarray(scores), np.asarray(groups)
    values = {}
    for task in np.unique(groups):
        selected = groups == task
        if len(np.unique(y[selected])) == 2:
            values[str(task)] = float(roc_auc_score(y[selected], scores[selected]))
    return values


def within_task_auc(y, scores, groups) -> float | None:
    values = list(task_auc(y, scores, groups).values())
    return float(np.mean(values)) if values else None


def predictive_metrics(y, scores, groups) -> dict:
    weights = task_weights(groups)
    return {'roc_auc': float(roc_auc_score(y, scores, sample_weight=weights)) if len(np.unique(y)) == 2 else None,
            'pr_auc': float(average_precision_score(y, scores, sample_weight=weights)) if np.any(y) else None,
            'log_loss': float(log_loss(y, scores, labels=[0, 1], sample_weight=weights)),
            'brier': float(brier_score_loss(y, scores, sample_weight=weights)),
            'within_task_auc': within_task_auc(y, scores, groups)}


def paired_task_bootstrap(values, *, replications: int, seed: int) -> dict:
    """One row per task; caller must form paired deltas before resampling."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError('Bootstrap requires a finite vector with exactly one contribution per task')
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(replications)])
    return {'estimate': float(values.mean()), 'ci95': np.quantile(means, [0.025, 0.975]).tolist(),
            'tasks': len(values), 'replications': replications, 'seed': seed, 'unit': 'task_id'}


def permute_within_tasks(y, groups, rng) -> np.ndarray:
    result = np.asarray(y).copy()
    groups = np.asarray(groups)
    for task in np.unique(groups):
        mask = groups == task
        result[mask] = rng.permutation(result[mask])
    return result


def within_task_permutation(y, baseline, combined, groups, *, replications: int, seed: int) -> dict:
    """Historical delta randomization; not a conditional incremental-value test.

    Retained to reproduce legacy artifacts. New analyses use fold_association_test.
    """
    # Precompute tie-aware pair comparisons. Only labels change in each randomization.
    y, baseline, combined, groups = map(np.asarray, (y, baseline, combined, groups))
    pairs = []
    for task in np.unique(groups):
        mask = groups == task
        labels = y[mask]
        if len(np.unique(labels)) != 2:
            continue
        def ranks(scores):
            return (scores[:, None] > scores[None, :]).astype(float) + 0.5 * (scores[:, None] == scores[None, :])
        delta = ranks(combined[mask]) - ranks(baseline[mask])
        pairs.append((labels, delta))
    if not pairs:
        return {'p_value': None, 'reason': 'No mixed tasks', 'replications': replications}
    def statistic(labels, matrix):
        return float(matrix[np.ix_(labels == 1, labels == 0)].mean())
    observed = np.mean([statistic(labels, matrix) for labels, matrix in pairs])
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(replications):
        null = np.mean([statistic(rng.permutation(labels), matrix) for labels, matrix in pairs])
        exceed += null >= observed
    return {'observed_delta': float(observed), 'p_value': (1 + exceed) / (replications + 1),
            'replications': replications, 'seed': seed, 'unit': 'task_id', 'alternative': 'greater'}


def fold_association_test(y, scores, groups, folds, *, replications: int, seed: int, n_folds: int) -> dict:
    """Conditional association in held-out folds; Bonferroni global-null test.

    A fold's scorer must not depend on any label in that fold. This tests
    intratask association, not conditional incremental information beyond B.
    """
    y, scores, groups, folds = map(np.asarray, (y, scores, groups, folds))
    for task in np.unique(groups):
        if len(np.unique(folds[groups == task])) != 1:
            raise ValueError('A task appears in multiple outer folds')
    if set(folds) - set(range(n_folds)):
        raise ValueError('Unexpected outer fold identifier')
    results = []
    for fold in range(n_folds):
        selected = folds == fold
        yf, sf, gf = y[selected], scores[selected], groups[selected]
        # Subtracting the constant tied-score AUC (0.5) leaves the association
        # randomization unchanged; it does not introduce observable baseline B.
        test = within_task_permutation(yf, np.full(len(yf), 0.5), sf, gf,
                                       replications=replications, seed=seed + fold)
        results.append({'fold': fold, 'p_value': test['p_value'] if test['p_value'] is not None else 1.0,
                        'observed_wt_auc': within_task_auc(yf, sf, gf),
                        'mixed_tasks': len(task_auc(yf, sf, gf)), 'seed': seed + fold})
    return {'p_value': min(1.0, n_folds * min(float(row['p_value']) for row in results)),
            'folds': results, 'combination': 'Bonferroni: min(1, K * min(p_k))',
            'null': 'No within-task score/outcome association in every outer fold',
            'not_a_test_of': 'Incremental information beyond observable features',
            'replications_per_fold': replications, 'alternative': 'greater',
            'assumption': 'Within-task label exchangeability conditional on the held-out fold scorer'}
