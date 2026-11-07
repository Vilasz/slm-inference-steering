"""Independent numerical and identity checks for the post-hoc reporting layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research_diagnostics import (
    calibration, common_cohort, contrasts, decision_ledger, influence,
    recall_evaluation, task_metrics, validate_predictions,
)


def predictions():
    # Unequal task sizes deliberately distinguish task weighting from candidate weighting.
    return pd.DataFrame([
        {'task_id': 'a', 'attempt_id': 0, 'checkpoint': '32', 'passed': 0, 'B': .2, 'H': .3, 'HB': .8, 'fold': 0},
        {'task_id': 'a', 'attempt_id': 1, 'checkpoint': '32', 'passed': 1, 'B': .8, 'H': .7, 'HB': .2, 'fold': 0},
        {'task_id': 'b', 'attempt_id': 0, 'checkpoint': '32', 'passed': 1, 'B': .6, 'H': .6, 'HB': .9, 'fold': 1},
    ])


def test_paired_task_losses_homogeneous_auc_and_influence():
    metrics = task_metrics(predictions())
    rows = contrasts(metrics, replications=100, seed=4)
    by_metric = {r['metric']: r['effect'] for r in rows if r['comparison'] == 'HB minus B'}
    assert by_metric['within_task_auc']['estimate'] == -1
    assert by_metric['within_task_auc']['tasks'] == 1  # homogeneous task excluded only from WT-AUC
    expected = (np.log(.8 / .2) + np.log(.6 / .9)) / 2
    assert by_metric['log_loss']['estimate'] == pytest.approx(expected)
    assert by_metric['log_loss']['tasks'] == 2
    assert rows == contrasts(metrics.sample(frac=1, random_state=3), replications=100, seed=4)
    changes = [r for r in influence(metrics) if r['metric'] == 'log_loss']
    assert next(r for r in changes if r['task_id'] == 'a')['leave_one_task_out_delta'] == pytest.approx(np.log(.6 / .9))


def test_calibration_equal_task_mass_and_probability_endpoints():
    frame = predictions()
    result = calibration(frame, 'B', bins=1, replications=100, seed=2)
    # Task a: success .5 / probability .5; task b: success 1 / probability .6.
    assert result['ece'] == pytest.approx(.2)
    assert result['rmsce'] == pytest.approx(.2)
    assert result['reliability'][0]['observed_rate'] == pytest.approx(.75)
    assert result == calibration(frame, 'B', bins=1, replications=100, seed=2)
    frame['B'] = frame.passed.astype(float)
    perfect = calibration(frame, 'B', replications=100)
    assert perfect['ece'] == 0 and perfect['ece_ci95'] == [0, 0]
    assert perfect['reliability'][-1]['mean_probability'] == 1
    assert sum(r['task_weight_mass'] for r in perfect['reliability']) == pytest.approx(1)


def test_common_cohort_preserves_attempt_identity_and_labels():
    frame = predictions()
    early = frame.assign(checkpoint='8')
    late = frame.iloc[[1, 2]].assign(checkpoint='64')
    selected = common_cohort(pd.concat([early, late]), checkpoints=('8', '64'))
    assert len(selected) == 4
    assert set(zip(selected.task_id, selected.attempt_id)) == {('a', 1), ('b', 0)}
    assert selected.groupby(['task_id', 'attempt_id']).passed.nunique().eq(1).all()
    assert selected.groupby('checkpoint').size().eq(2).all()
    assert common_cohort(early).empty


@pytest.mark.parametrize('corruption', ['duplicate', 'nonfinite', 'range', 'label', 'fold', 'eligibility'])
def test_invalid_frozen_prediction_tables_rejected(corruption):
    frame = predictions()
    if corruption == 'duplicate':
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif corruption == 'nonfinite':
        frame.loc[0, 'HB'] = np.inf
    elif corruption == 'range':
        frame.loc[0, 'HB'] = -.01
    elif corruption == 'label':
        other = frame.iloc[[0]].assign(checkpoint='8', passed=1)
        frame = pd.concat([frame, other])
    elif corruption == 'fold':
        frame.loc[1, 'fold'] = 1
    else:
        frame['generated_tokens'] = 32
    with pytest.raises(ValueError):
        validate_predictions(frame)


def test_recall_uses_all_active_correct_candidates_and_frozen_threshold_ties():
    frame = predictions()
    policies = [{'policy': 'HB', 'checkpoint': 32, 'threshold': .9, 'validation_correct_recall': .98}]
    result = recall_evaluation(frame, policies, replications=200, seed=5)
    aggregate = result['aggregate'][0]
    assert aggregate['active_correct'] == 2  # both correct candidates, regardless of replay initiation
    assert aggregate['retained_correct'] == 1  # score == threshold continues
    assert aggregate['recall'] == .5
    original_scores = frame[['B', 'H', 'HB']].copy()
    original_policies = [dict(p) for p in policies]
    changed = frame.copy()
    changed['passed'] = 1 - changed.passed
    recall_evaluation(changed, policies, replications=20)
    pd.testing.assert_frame_equal(frame[['B', 'H', 'HB']], original_scores)
    assert policies == original_policies
    with pytest.raises(ValueError, match='one selected gate'):
        recall_evaluation(frame, policies * 2, replications=10)


def test_recall_no_correct_candidates_is_unidentified():
    frame = predictions().assign(passed=0)
    result = recall_evaluation(frame, [{'policy': 'HB', 'checkpoint': '32', 'threshold': .5,
                                     'validation_correct_recall': .97}], replications=10)
    assert result['aggregate'][0]['recall'] is None
    assert result['aggregate'][0]['ci95'] is None
    assert result['aggregate'][0]['undefined_bootstrap_draws'] == 10


def test_decision_ledger_preserves_strict_frozen_boundaries():
    main = {'signal': {'H2': {'delta_hb_minus_b': {'estimate': 0, 'ci95': [0, .1]},
                            'delta_log_loss_hb_minus_b': {'estimate': .03, 'ci95': [.01, .05]}, 'supported': False}},
            'policies': {'comparisons': [{'budget': 20, 'comparison': 'HB minus verifier',
                                         'resolution_delta': {'estimate': 0, 'ci95': [-.03, .01]},
                                         'token_delta': {'estimate': -1, 'ci95': [-2, 0]}}],
                         'H3': {'operational_benefit_claim_supported': False}}}
    rows = decision_ledger(main, {'passed': True, 'has_actual_abort': True})
    assert all(not r['satisfied'] for r in rows if r['criterion'] != 'Online identity replay with actual abort')
    assert next(r for r in rows if r['criterion'] == 'Online identity replay with actual abort')['satisfied']


def test_publication_rejects_reinterpreted_primary_results():
    from scripts.audit_research_delivery import audit_supplemental
    primary = {'estimate': -.04, 'ci95': [-.1, .01]}
    loss = {'estimate': .03, 'ci95': [.01, .06]}
    extra = {'status': 'post_hoc_exploratory_frozen_predictions', 'artifacts': {'table.csv': 'abc'},
             'primary_reproduction': {'within_task_auc': {'matches': True, 'effect': dict(primary)},
                                      'log_loss': {'matches': True, 'effect': dict(loss)}},
             'decision_ledger': [{'criterion': 'H2 joint criterion', 'satisfied': False},
                                 {'criterion': 'Joint operational benefit', 'satisfied': False}]}
    evidence = {'supplemental': extra, 'inputs': {'table.csv': 'abc'},
                'main': {'signal': {'H2': {'delta_hb_minus_b': primary,
                                         'delta_log_loss_hb_minus_b': loss, 'supported': False}},
                         'policies': {'H3': {'operational_benefit_claim_supported': False}}}}
    assert audit_supplemental(evidence)['original_decisions_preserved']
    extra['primary_reproduction']['log_loss']['effect'] = {'estimate': -.03, 'ci95': [-.06, -.01]}
    with pytest.raises(ValueError, match='differs from the frozen'):
        audit_supplemental(evidence)


def test_ledger_log_loss_condition_uses_the_original_point_rule():
    main = {'signal': {'H2': {'delta_hb_minus_b': {'estimate': .03, 'ci95': [.01, .05]},
                            'delta_log_loss_hb_minus_b': {'estimate': -.01, 'ci95': [-.04, .02]}, 'supported': True}},
            'policies': {'comparisons': [], 'H3': {'operational_benefit_claim_supported': False}}}
    row = next(r for r in decision_ledger(main, {}) if r['criterion'] == 'Primary log-loss reduction')
    assert row['satisfied'] is True
    assert row['direction'] == 'negative_point'
