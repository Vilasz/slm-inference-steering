from __future__ import annotations

import numpy as np
import pandas as pd

from slm_steering.research.policy import replay
from slm_steering.research.protocol import ROOT, load_config
from slm_steering.research.reporting import summarize_policies, summarize_signal


def test_incremental_decision_is_distinct_from_fold_association():
    config = load_config(ROOT / 'configs/protocol_v0.1.yaml')
    config['statistics'].update(bootstrap_replications=50, permutation_replications=99)
    rows = []
    for task in range(12):
        for attempt in range(4):
            y = attempt % 2
            rows.append({'checkpoint': '32', 'task_id': f'task{task}', 'fold': task % 5, 'passed': y, 'truncated': False,
                         'prior': 0.5, 'B1': 0.5, 'B2': 0.5, 'B3': 0.5, 'B': 0.5,
                         'H': 0.8 if y else 0.2, 'HB': 0.8 if y else 0.2,
                         'random_direction': 0.5, 'shuffled_training_labels': 0.5, 'length_stopping': 0.5,
                         'hidden_linear_score': (y * 2 - 1) * (0.5 + task / 20), 'observable_0': 1.0})
    result = summarize_signal(pd.DataFrame(rows), config)
    assert result['H2']['supported']
    assert result['H2']['delta_hb_minus_b']['ci95'] == [0.5, 0.5]
    # With 99 permutations/fold, Bonferroni's minimum attainable p is 0.05.
    # This association test must not be repurposed as an incremental-gain gate.
    assert result['H2']['permutation']['p_value'] >= 0.05
    assert result['risk_sets'][0]['mixed_tasks'] == 12
    # Ranking alone does not suffice if combined predictions worsen log loss.
    for row in rows:
        row['HB'] = 0.09 if row['passed'] else 0.01
    bad_loss = summarize_signal(pd.DataFrame(rows), config)
    assert not bad_loss['H2']['supported']


def test_h3_offline_support_does_not_finalize_without_online_validation():
    config = load_config(ROOT / 'configs/protocol_v0.1.yaml')
    config['statistics']['bootstrap_replications'] = 50
    records = [{'task_id': f'task{task}', 'attempt_id': attempt, 'seed': task * 100 + attempt,
                'passed': attempt == 1, 'generated_tokens': 100} for task in range(10) for attempt in range(20)]
    scores = {(r['task_id'], r['attempt_id']): 0.9 if r['passed'] else 0.1 for r in records}
    rows = []
    for budget in [1, 5, 10, 20]:
        for name in ['HB', 'B', 'verifier', 'fixed']:
            replayed = replay(records, scores, checkpoint=32, threshold=0.5, budget=budget,
                              policy='adaptive' if name in ['HB', 'B'] else name)
            rows.extend([{**row, 'policy': name} for row in replayed])
    result = summarize_policies(pd.DataFrame(rows), config)
    assert result['H3']['offline_statistical_support']
    assert not result['H3']['system_result_finalized']
    online = summarize_policies(pd.DataFrame(rows), config, online_validated=True)
    assert not online['H3']['incremental_over_observables']
    assert not online['H3']['system_result_finalized']


def test_unequal_risk_set_counts_still_weight_each_task_equally():
    from slm_steering.research.statistics import predictive_metrics
    y = np.array([1, 0] + [1, 0] * 10)
    probabilities = np.array([0.9, 0.1] + [0.1, 0.9] * 10)
    groups = np.array(['a'] * 2 + ['b'] * 20)
    metrics = predictive_metrics(y, probabilities, groups)
    assert metrics['within_task_auc'] == 0.5
    assert abs(metrics['log_loss'] - (-np.log(0.9) - np.log(0.1)) / 2) < 1e-10


def test_artifact_export_produces_valid_json_tables_and_figures(tmp_path):
    import json
    from slm_steering.research.reporting import write_artifacts
    config = load_config(ROOT / 'configs/protocol_v0.1.yaml')
    config['statistics'].update(bootstrap_replications=10, permutation_replications=9)
    rows = []
    for task in range(3):
        for attempt in range(4):
            y = attempt % 2
            rows.append({'checkpoint': '32', 'task_id': f'task{task}', 'fold': task % 5, 'passed': y, 'truncated': False,
                         'prior': 0.5, 'B1': 0.5, 'B2': 0.5, 'B3': 0.5, 'B': 0.5,
                         'H': 0.7 if y else 0.3, 'HB': 0.7 if y else 0.3,
                         'random_direction': 0.5, 'shuffled_training_labels': 0.5, 'length_stopping': 0.5,
                         'hidden_linear_score': y - 0.5, 'observable_0': 1.0})
    write_artifacts(pd.DataFrame(rows), pd.DataFrame(), config, tmp_path)
    assert json.loads((tmp_path / 'results.json').read_text())['policies']['status'] == 'not_run'
    assert (tmp_path / 'tables/risk_sets.csv').exists()
    assert (tmp_path / 'figures/temporal_prediction.png').stat().st_size > 1000
    assert (tmp_path / 'figures/within_task_discrimination.png').stat().st_size > 1000
    assert (tmp_path / 'figures/calibration_32.png').stat().st_size > 1000
    for name in ('risk_set_attrition', 'primary_predictor_comparison', 'primary_task_deltas', 'primary_roc_pr'):
        for suffix in ('png', 'svg', 'pdf'):
            assert (tmp_path / f'figures/{name}.{suffix}').stat().st_size > 1000
