import numpy as np
import pandas as pd
import pytest

from slm_steering.excellence.independent_report import calibration_rows, deletion_sensitivity, recall_intervals


def test_recall_uses_candidate_counts_instead_of_mean_of_task_ratios():
    rows = recall_intervals([[1, 2, 2], [0, 1, 1]], replications=1000, seed=42)
    hb, baseline = rows
    assert hb['recall'] == pytest.approx(1 / 3)
    assert hb['recall'] != pytest.approx((.5 + 0.) / 2)
    assert baseline['recall'] == 1.
    assert baseline['ci95'] == [1., 1.]
    assert hb['active_correct'] == baseline['active_correct'] == 3
    assert hb['undefined_draws'] == 0


def test_retention_without_any_correct_candidate_stays_undefined():
    for row in recall_intervals(np.zeros((3, 3)), replications=400, seed=42):
        assert row['recall'] is None and row['ci95'] is None
        assert row['undefined_draws'] == 400


def test_calibration_preserves_task_weights_and_includes_score_one():
    frame = pd.DataFrame({'task_id': ['a', 'b', 'b', 'b'],
                          'passed': [0, 0, 1, 1], 'B': [0., 1., 1., 1.]})
    rows, ece = calibration_rows(frame, 'B')
    assert len(rows) == 10 and rows[9]['candidates'] == 3
    assert rows[0]['weight'] == pytest.approx(.5)
    assert rows[9]['weight'] == pytest.approx(.5)
    assert ece == pytest.approx(1 / 6)
    assert all(row['mean_score'] is None for row in rows[1:9])


def test_deletion_sensitivity_recomputes_the_ratio_and_keeps_undefined_cases():
    values = deletion_sensitivity([[10, 10, 10], [0, 10, 10]])
    assert [v['estimate_without_task'] for v in values] == [-1., 0.]
    assert [v['change_from_full'] for v in values] == [-.5, .5]
    assert deletion_sensitivity([[0, 0, 0]]) == [{'estimate_without_task': None, 'change_from_full': None}]
def test_independent_namespace_keeps_official_functional_label_contract():
    from slm_steering.excellence.independent_report import validate_functional_record
    import pytest
    record = {'passed': True, 'verifier_status': {'base': 'pass', 'plus': 'pass'},
              'label_definition': 'evalplus_base_and_plus_pass', 'error_type': None, 'token_margins': [0., 1.]}
    validate_functional_record(record)
    with pytest.raises(ValueError, match='functional label'):
        validate_functional_record({**record, 'verifier_status': {'base': 'pass', 'plus': 'fail'}})
    with pytest.raises(ValueError, match='margins'):
        validate_functional_record({**record, 'token_margins': [None]})

