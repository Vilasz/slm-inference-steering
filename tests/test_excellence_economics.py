from __future__ import annotations

import numpy as np
import pytest

from slm_steering.excellence.economics import checkpoint_policy, expected_cost, simulate


def stream():
    return [{'task_id': 'task', 'attempt_id': i, 'generated_tokens': length, 'passed': passed}
            for i, (length, passed) in enumerate([(8, False), (100, False), (60, True), (12, True)])]


def test_oracle_obeys_termination_and_preserves_first_success():
    records = stream()
    oracle = simulate(records, checkpoint=8, budget=4, kind='oracle')
    verifier = simulate(records, checkpoint=8, budget=4, kind='verifier')
    assert oracle['tokens'] == 8 + 8 + 60
    assert verifier['tokens'] == 8 + 100 + 60
    assert oracle['resolved'] == verifier['resolved'] == 1
    assert oracle['started'] == verifier['started'] == 3
    assert oracle['false_aborts'] == 0
    assert oracle['decisions'][0]['aborted'] is False  # natural finish at t


def test_false_abort_does_not_resolve_or_start_unneeded_candidates():
    records = stream()
    scores = {('task', i): p for i, p in enumerate([.1, .1, .1, .9])}
    gate = simulate(records, checkpoint=8, threshold=.5, scores=scores, budget=4, kind='gate')
    assert gate['started'] == 4 and gate['resolved'] == 1
    assert gate['tokens'] == 8 + 8 + 8 + 12
    assert gate['false_aborts'] == 1 and gate['active_correct_started'] == 2
    # A success-discarding policy can cost less than the all-success-preserving oracle.
    assert gate['tokens'] < simulate(records, checkpoint=8, budget=4, kind='oracle')['tokens']
    assert simulate(records, checkpoint=8, budget=3, threshold=.5, scores=scores, kind='gate')['resolved'] == 0


def test_finite_budget_cost_formula_matches_enumeration():
    # Cost depends on the current success label, but current attempt is independent of previous attempts.
    p = .3
    mean = p * 50 + (1 - p) * 8
    exact = 0.
    for bits in np.ndindex(*(2,) * 3):
        probability = np.prod([p if b else 1 - p for b in bits])
        cost = 0
        for b in bits:
            cost += 50 if b else 8
            if b:
                break
        exact += probability * cost
    assert expected_cost(mean, p, 3) == pytest.approx(exact)
    assert expected_cost(7, 0, 3) == 21
    assert expected_cost(7, 1, 3) == 7


def test_checkpoint_threshold_selection_uses_development_cost_not_target_labels():
    options = [{'checkpoint': 8, 'mean_tokens': 20., 'resolution': .8, 'threshold': .4},
               {'checkpoint': 8, 'mean_tokens': 20., 'resolution': .9, 'threshold': .3},
               {'checkpoint': 32, 'mean_tokens': 1., 'resolution': 1., 'threshold': .9}]
    assert checkpoint_policy(options, 8)['threshold'] == .3
    with pytest.raises(ValueError, match='No source-development'):
        checkpoint_policy(options, 64)


@pytest.mark.parametrize('lengths,labels,oracle_cost,verifier_cost,started,resolved', [
    ([5, 20, 30], [False, False, False], 21, 55, 3, 0),
    ([12, 40, 9], [True, False, True], 12, 12, 1, 1),
    ([20, 30, 12], [False, False, True], 28, 62, 3, 1),
    ([8, 40, 20], [True, False, True], 8, 8, 1, 1),
])
def test_oracle_manual_cases(lengths, labels, oracle_cost, verifier_cost, started, resolved):
    records = [{'task_id': 'manual', 'attempt_id': i, 'generated_tokens': n, 'passed': y}
               for i, (n, y) in enumerate(zip(lengths, labels))]
    oracle = simulate(records, checkpoint=8, budget=3, kind='oracle')
    verifier = simulate(records, checkpoint=8, budget=3, kind='verifier')
    assert oracle['tokens'] == oracle_cost and verifier['tokens'] == verifier_cost
    assert oracle['resolved'] == verifier['resolved'] == resolved
    assert oracle['started'] == verifier['started'] == started
    assert oracle['false_aborts'] == 0


def test_oracle_coverage_and_cost_on_every_binary_stream_and_budget():
    for labels in np.ndindex(*(2,) * 4):
        records = [{'task_id': 'enumeration', 'attempt_id': i, 'generated_tokens': n, 'passed': bool(y)}
                   for i, (n, y) in enumerate(zip([8, 9, 20, 1], labels))]
        for budget in (1, 2, 3, 4):
            oracle = simulate(records, checkpoint=8, budget=budget, kind='oracle')
            verifier = simulate(records, checkpoint=8, budget=budget, kind='verifier')
            assert oracle['resolved'] == verifier['resolved']
            assert oracle['started'] == verifier['started']
            assert oracle['tokens'] <= verifier['tokens']
            assert all(not decision['aborted'] for decision in oracle['decisions']
                       if records[decision['attempt_id']]['passed'])


def test_gate_does_not_use_future_label_or_suffix_length_to_abort():
    scores = {('manual', 0): .1, ('manual', 1): .9}
    original = [{'task_id': 'manual', 'attempt_id': 0, 'generated_tokens': 20, 'passed': True},
                {'task_id': 'manual', 'attempt_id': 1, 'generated_tokens': 12, 'passed': True}]
    changed = [dict(original[0], generated_tokens=200, passed=False), original[1]]
    a = simulate(original, checkpoint=8, budget=2, threshold=.5, scores=scores, kind='gate')
    b = simulate(changed, checkpoint=8, budget=2, threshold=.5, scores=scores, kind='gate')
    assert a['decisions'] == b['decisions']
    assert a['tokens'] == b['tokens'] == 20
    assert a['started'] == b['started'] == 2 and a['resolved'] == b['resolved'] == 1


def test_natural_success_at_checkpoint_never_needs_a_probe_score():
    records = [{'task_id': 'terminal', 'attempt_id': 0, 'generated_tokens': 8, 'passed': True},
               {'task_id': 'terminal', 'attempt_id': 1, 'generated_tokens': 512, 'passed': False}]
    gate = simulate(records, checkpoint=8, budget=2, threshold=.99, scores={}, kind='gate')
    assert gate['tokens'] == 8 and gate['started'] == 1 and gate['resolved'] == 1
    assert gate['decisions'] == [{'attempt_id': 0, 'aborted': False, 'tokens': 8}]


def test_cost_formula_with_correct_and_incorrect_aborts_matches_full_enumeration():
    # Outcomes: kept correct, aborted correct, kept incorrect, aborted incorrect.
    probabilities = [.3, .1, .15, .45]
    costs = [50, 8, 40, 8]
    exact = 0.
    for outcomes in np.ndindex(4, 4, 4):
        probability = float(np.prod([probabilities[o] for o in outcomes]))
        cost = 0
        for outcome in outcomes:
            cost += costs[outcome]
            if outcome == 0:
                break
        exact += probability * cost
    mean_cost = float(np.dot(probabilities, costs))
    assert expected_cost(mean_cost, .4 * .75, 3) == pytest.approx(exact)
    assert 1 - (1 - .4 * .75) ** 3 < 1 - (1 - .4) ** 3
