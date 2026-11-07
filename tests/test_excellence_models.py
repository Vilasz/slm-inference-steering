from __future__ import annotations

import numpy as np

from slm_steering.excellence.models import (
    Projection, ResidualProcedure, fit_procedure, fitted_predict, grouped_splits, offset_fit,
)


def synthetic_data():
    rng = np.random.default_rng(134)
    groups = np.repeat([f't{i:02d}' for i in range(18)], 6)
    y = np.tile([0, 1, 0, 1, 0, 1], 18)
    obs = rng.normal(size=(len(y), 10))
    obs[:, 5] += y
    return {'groups': groups, 'y': y, 'observable': obs,
            'hidden': rng.normal(size=(len(y), 1, 8)), 'layers': [3]}


def test_projection_is_fit_only_on_training_rows():
    data = synthetic_data()
    train, test = np.arange(72), np.arange(72, 108)
    projection = Projection({'mode': 'combined', 'layer': 3, 'components': 2}, [3]).fit(data, train)
    original = projection.transform(data, train).copy()
    data['hidden'][test] = 1e6
    data['observable'][test] = -1e6
    np.testing.assert_array_equal(projection.transform(data, train), original)
    assert not set(projection.fit_tasks) & set(data['groups'][test])
    assert projection.pca.n_samples_ == len(train)


def test_whole_selection_calibration_excludes_outer_labels():
    data = synthetic_data()
    train, test = np.arange(72), np.arange(72, 108)
    variants = [{'mode': 'observable', 'columns': [5]}]
    first = fit_procedure(data, train, variants, [.01, .1])
    p = fitted_predict(first, data, test)
    data['y'][test] = 1 - data['y'][test]
    second = fit_procedure(data, train, variants, [.01, .1])
    np.testing.assert_array_equal(fitted_predict(second, data, test), p)
    for row in first['calibration_audit']:
        assert not set(row['fit_tasks']) & set(row['score_tasks'])
        assert not (set(row['fit_tasks']) | set(row['score_tasks'])) & set(data['groups'][test])


def test_residual_offsets_exclude_their_labels_including_calibrator():
    data = synthetic_data()
    indices = np.arange(len(data['y']))
    _, score = next(grouped_splits(data, indices))
    original, audit = ResidualProcedure(data).honest_offsets(indices)
    changed = {**data, 'y': data['y'].copy()}
    changed['y'][score] = 1 - changed['y'][score]
    revised, _ = ResidualProcedure(changed).honest_offsets(indices)
    np.testing.assert_array_equal(original[score], revised[score])
    for row in audit:
        assert not set(row['first_stage_fit_tasks']) & set(row['offset_score_tasks'])


def test_zero_offset_signal_cannot_create_spurious_correction():
    data = synthetic_data()
    x = np.zeros((len(data['y']), 4))
    w = offset_fit(x, data['y'], np.zeros(len(x)), data['groups'], .001)
    np.testing.assert_array_equal(w, np.zeros(4))
