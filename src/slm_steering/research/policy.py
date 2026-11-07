from __future__ import annotations

import numpy as np


def recall_threshold(y, probabilities, minimum_recall: float = 0.97) -> float:
    """Largest observed recall-admissible threshold; not a cost optimizer."""
    positive = np.sort(np.asarray(probabilities)[np.asarray(y) == 1])
    if not len(positive):
        raise ValueError('No correct validation trajectories; recall threshold cannot be selected')
    allowed = int(np.floor((1 - minimum_recall) * len(positive) + 1e-10))
    return float(positive[allowed])


def replay(records: list[dict], probabilities: dict, *, checkpoint: int, threshold: float, budget: int,
           policy: str = 'adaptive') -> list[dict]:
    if budget < 1 or checkpoint < 1 or policy not in {'adaptive', 'verifier', 'fixed'}:
        raise ValueError('Invalid replay budget, checkpoint or policy')
    result = []
    by_task: dict[str, list] = {}
    for record in records:
        by_task.setdefault(record['task_id'], []).append(record)
    for task, stream in sorted(by_task.items()):
        attempts = sorted(stream, key=lambda r: r['attempt_id'])[:budget]
        if len(attempts) != budget:
            raise ValueError(f'{task} has fewer attempts than N={budget}; incomplete streams cannot be compared')
        cost, full, solved, false_aborts, positives, active_positives = 0, 0, False, 0, 0, 0
        identities, decisions = [], []
        for record in attempts:
            identity = (task, record['attempt_id'])
            identities.append({'task_id': task, 'attempt_id': record['attempt_id'], 'seed': record['seed']})
            eligible = record['generated_tokens'] > checkpoint
            probability = None
            if policy == 'adaptive' and eligible:
                if identity not in probabilities:
                    raise ValueError('Missing risk-set score; never silently retain an unscored eligible attempt')
                probability = float(probabilities[identity])
                if not np.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError('Policy probabilities must be finite and in [0, 1]')
            abort = probability is not None and probability < threshold
            # The action above depends only on active status and a prefix score.
            positives += bool(record['passed'])
            active_positives += bool(record['passed']) and eligible
            charged = checkpoint if abort else record['generated_tokens']
            cost += charged
            decisions.append({'attempt_id': record['attempt_id'], 'active_at_checkpoint': eligible,
                              'probability': probability, 'aborted': abort, 'completion_tokens': charged})
            if abort:
                false_aborts += bool(record['passed'])
            else:
                full += 1
                solved |= bool(record['passed'])
                if solved and policy != 'fixed':
                    break
        result.append({'task_id': task, 'policy': policy, 'budget': budget, 'completion_tokens': cost,
                       'attempts_started': len(identities), 'fully_generated_attempts': full, 'resolved': int(solved),
                       'correct_false_aborts': false_aborts, 'correct_attempts_started': positives,
                       'active_correct_attempts_started': active_positives,
                       'attempt_identities': identities, 'decisions': decisions})
    return result


def select_policy(records: list[dict], candidates: list[dict], *, minimum_recall: float, budget: int,
                  tie_tolerance: float = 1e-9) -> dict:
    """Search every empirical strict-below decision boundary, including no abort."""
    if not candidates or not 0 < minimum_recall <= 1:
        raise ValueError('Policy selection needs candidates and a recall target in (0, 1]')
    options = []
    for candidate in candidates:
        checkpoint, lookup = candidate['checkpoint'], candidate['probabilities']
        eligible = [r for r in records if r['generated_tokens'] > checkpoint]
        y = np.array([r['passed'] for r in eligible], dtype=bool)
        probabilities = np.array([lookup[(r['task_id'], r['attempt_id'])] for r in eligible])
        if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError('Policy probabilities must be finite and in [0, 1]')
        grid = [0.0]
        if y.any():
            grid.extend(np.nextafter(np.unique(probabilities), np.inf).tolist())
        for threshold in grid:
            recall = float(np.mean(probabilities[y] >= threshold)) if y.any() else None
            if recall is not None and recall + 1e-12 < minimum_recall:
                continue
            rows = replay(records, lookup, checkpoint=checkpoint, threshold=threshold, budget=budget)
            options.append({'checkpoint': checkpoint, 'threshold': float(threshold),
                            'mean_tokens': float(np.mean([r['completion_tokens'] for r in rows])),
                            'resolution': float(np.mean([r['resolved'] for r in rows])),
                            'validation_correct_recall': recall, 'active_validation_positives': int(y.sum()),
                            'no_abort': threshold == 0.0})
    minimum_cost = min(option['mean_tokens'] for option in options)
    tied = [option for option in options if abs(option['mean_tokens'] - minimum_cost) <= tie_tolerance]
    selected = min(tied, key=lambda option: (-option['resolution'], option['checkpoint'], option['threshold']))
    return {**selected, 'candidates': options, 'selection_data': 'outer_training_development_only',
            'recall_interpretation': 'Empirical development target, not an external guarantee',
            'tie_tolerance': tie_tolerance}
