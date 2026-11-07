from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from slm_steering.research.bank import select_temperature
from slm_steering.research.features import observable_features
from slm_steering.research.policy import recall_threshold, replay
from slm_steering.research.protocol import (ROOT, attempt_seed, block_indices, canonical, digest, load_config,
                                          pilot_task_ids, require_disjoint, token_position, validate_bank)
from slm_steering.research.statistics import paired_task_bootstrap, permute_within_tasks, within_task_auc, within_task_permutation


def record(task='task', attempt=0, passed=False, length=40):
    return {'protocol_version': '0.1', 'run_id': 'fixture', 'benchmark_name': 'fixture', 'benchmark_version': 'v1',
            'task_id': task, 'attempt_id': attempt, 'seed': attempt_seed('0.1', 'v1', task, attempt),
            'model_id': 'fixture', 'model_revision': 'a' * 40, 'tokenizer_revision': 'a' * 40,
            'prompt_template_hash': 'a' * 64, 'decoding_config': {'temperature': 0.8},
            'prompt': 'def f():\n    """Example."""', 'raw_completion': 'return 1', 'extracted_code': 'return 1',
            'verifier_version': 'fixture', 'passed': passed, 'error_type': None if passed else 'evalplus_fail',
            'generated_tokens': length, 'finish_reason': 'eos', 'truncated': False,
            'generation_time_seconds': 1.0, 'verification_time_seconds': 0.1,
            'token_ids': list(range(length)), 'token_logprobs': [-0.5] * length,
            'token_entropies': [1.0] * length, 'token_margins': [0.1] * length, 'prompt_token_ids': [1, 2, 3],
            'signature_token_count': 3, 'development_only': False,
            'artifact_hashes': {'completion': digest('return 1'), 'code': digest('return 1'),
                                'tokens': digest(canonical(list(range(length))))}}


def test_pilot_selection_is_hash_based_and_order_invariant():
    config = load_config(ROOT / 'configs/protocol_v0.1.yaml')
    tasks = [f'HumanEval/{i}' for i in range(164)]
    expected = sorted(tasks, key=lambda task: digest('protocol-v0.1-pilot:' + task))[:32]
    assert pilot_task_ids(tasks[::-1], config) == expected
    assert len(set(expected)) == 32


def test_seeds_are_independent_of_attempt_execution_order():
    first = {(task, i): attempt_seed('0.1', 'v1', task, i) for task in ['a', 'b'] for i in range(20)}
    second = {(task, i): attempt_seed('0.1', 'v1', task, i) for task in ['b', 'a'] for i in reversed(range(20))}
    assert first == second
    assert len(set(first.values())) == 40


@pytest.mark.parametrize('checkpoint,position', [(0, 99), (8, 107), (16, 115), (32, 131), (64, 163), ('final', 179)])
def test_offsets_are_relative_to_completion(checkpoint, position):
    assert token_position(100, 80, checkpoint) == position


def test_short_completion_has_no_imputed_late_activation():
    with pytest.raises(ValueError):
        token_position(100, 12, 32)
    with pytest.raises(ValueError):
        observable_features(record(length=12), 32)


def test_layers_are_zero_based_block_outputs():
    assert block_indices(28, [0.25, 0.5, 0.75, 1.0]) == [6, 13, 20, 27]


def test_group_and_pilot_leakage_fail_closed():
    require_disjoint(['a'], ['b'], ['pilot'])
    with pytest.raises(ValueError, match='leakage'):
        require_disjoint(['a'], ['a'])
    with pytest.raises(ValueError, match='Pilot'):
        require_disjoint(['a'], ['pilot'], ['pilot'])


def test_bank_rejects_duplicate_ids_seeds_corruption_and_missing_provenance():
    first = record()
    validate_bank([first])
    with pytest.raises(ValueError, match='Duplicate'):
        validate_bank([first, first])
    missing = deepcopy(first)
    del missing['verifier_version']
    with pytest.raises(ValueError, match='provenance'):
        validate_bank([missing])
    broken = deepcopy(first)
    broken['token_ids'][0] = 999
    with pytest.raises(ValueError, match='Corrupt'):
        validate_bank([broken])
    broken = deepcopy(first)
    broken['token_logprobs'].pop()
    with pytest.raises(ValueError, match='length'):
        validate_bank([broken])
    broken = deepcopy(first)
    broken.update(truncated=True, passed=True)
    with pytest.raises(ValueError, match='Truncation'):
        validate_bank([broken])


def test_observable_prefix_features_are_invariant_to_future_statistics():
    first = record(length=40)
    second = deepcopy(first)
    for field in ['token_logprobs', 'token_entropies', 'token_margins']:
        second[field][32:] = [999] * 8
    np.testing.assert_array_equal(observable_features(first, 32), observable_features(second, 32))


def test_within_task_auc_balances_tasks_and_handles_ties():
    y = [1, 0, 1, 0, 0, 0]
    groups = ['a', 'a', 'b', 'b', 'b', 'b']
    assert within_task_auc(y, [1, 0, 0, 1, 1, 1], groups) == 0.5
    assert within_task_auc(y, [0.5] * 6, groups) == 0.5
    assert within_task_auc([1, 1], [0.9, 0.8], ['a', 'a']) is None


def test_permutation_preserves_every_tasks_prevalence():
    y = np.array([1, 0, 1, 0, 0, 1, 1, 1])
    groups = np.array(['a'] * 4 + ['b'] * 4)
    permuted = permute_within_tasks(y, groups, np.random.default_rng(4))
    for group in np.unique(groups):
        assert y[groups == group].sum() == permuted[groups == group].sum()
    result = within_task_permutation(y, [0.5] * 8, y, groups, replications=99, seed=4)
    assert result['observed_delta'] == 0.5
    assert 0 < result['p_value'] <= 1


def test_bootstrap_resamples_one_contribution_per_task():
    result = paired_task_bootstrap([1, 1, 1], replications=50, seed=4)
    assert result['unit'] == 'task_id'
    assert result['ci95'] == [1.0, 1.0]
    with pytest.raises(ValueError):
        paired_task_bootstrap(np.zeros((2, 3)), replications=10, seed=4)


def test_recall_threshold_and_strict_abort_tie_rule():
    positive = np.linspace(0, 1, 100)
    threshold = recall_threshold(np.ones(100), positive)
    assert np.mean(positive >= threshold) == 0.97
    assert recall_threshold([1, 1], [0.4, 0.4]) == 0.4
    with pytest.raises(ValueError):
        recall_threshold([0, 0], [0.1, 0.2])


def test_replay_exact_abort_cost_short_natural_finish_and_stream_identity():
    records = [record(attempt=0, length=50), record(attempt=1, length=8), record(attempt=2, passed=True, length=40)]
    scores = {('task', 0): 0.1, ('task', 2): 0.9}
    row = replay(records, scores, checkpoint=32, threshold=0.5, budget=3)[0]
    assert row['completion_tokens'] == 32 + 8 + 40
    assert row['fully_generated_attempts'] == 2
    assert row['resolved'] == 1
    assert [i['seed'] for i in row['attempt_identities']] == [r['seed'] for r in records]
    assert replay(records, {}, checkpoint=32, threshold=0, budget=3, policy='verifier')[0]['completion_tokens'] == 98
    with pytest.raises(ValueError, match='Missing'):
        replay(records, {}, checkpoint=32, threshold=0.5, budget=3)


def test_pilot_cannot_select_temperature_before_full_grid():
    config = load_config(ROOT / 'configs/protocol_v0.1.yaml')
    partial = select_temperature([], config, ['a', 'b'])
    assert not partial['complete']
    assert partial['selected_temperature'] is None


def test_definitive_generation_rejects_pending_protocol_approval(tmp_path, monkeypatch):
    from slm_steering.research import protocol
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    (tmp_path / 'RESEARCH_PROTOCOL.md').write_text('**Status:** draft for advisor approval\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Gate 0'):
        protocol.require_freeze(tmp_path / 'config.yaml', {})


def test_pilot_selection_follows_mixed_tasks_then_declared_tiebreakers():
    config = load_config(ROOT / 'configs/protocol_v0.1.yaml')
    config['pilot']['attempts_per_task'] = 2
    records = []
    for temperature in [0.6, 0.8, 1.0]:
        for task in ['a', 'b']:
            for attempt in range(2):
                item = record(task, attempt, passed=attempt == 0, length=40 if temperature == 0.8 else 50)
                item['decoding_config']['temperature'] = temperature
                records.append(item)
    selected = select_temperature(records, config, ['a', 'b'])
    assert selected['complete']
    assert selected['selected_temperature'] == 0.8


def test_sampling_statistics_match_huggingface_top_p_distribution():
    import importlib.util
    import subprocess
    import sys
    if importlib.util.find_spec('torch') is None:
        pytest.skip('torch not installed')
    # Isolate native torch and scipy OpenMP DLLs on Windows; do not mask errors.
    code = '''import torch
import sys
sys.path.insert(0, 'src')
from transformers import TopPLogitsWarper
from slm_steering.research.generation import sampling_statistics
logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])
probabilities, logprobs, entropy, margin = sampling_statistics(logits, 0.8, 0.95)
warped = TopPLogitsWarper(0.95)(torch.tensor([[0]]), logits / 0.8)
torch.testing.assert_close(probabilities, warped.softmax(-1))
assert float(entropy) > 0
assert torch.allclose(probabilities.sum(), torch.tensor(1.0))
assert torch.isneginf(logprobs[0, -1])
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr


def test_nested_predictions_never_train_on_outer_test_tasks():
    from slm_steering.research.probes import grouped_predictions
    config = load_config(ROOT / 'configs/protocol_v0.1.yaml')
    config['statistics'].update(regularization_c=[0.1], inner_folds=2)
    config['activations']['checkpoints'] = [0, 32]
    rng = np.random.default_rng(5)
    records = [record(f'task{task}', attempt, passed=attempt % 2 == 0) for task in range(10) for attempt in range(4)]
    hidden = rng.normal(size=(40, 1, 3))
    prompt = np.repeat(rng.normal(size=(10, 1, 3)), 4, axis=0)
    observable = np.array([observable_features(r, 32) for r in records])
    features = {str(t): {'indices': np.arange(40), 'hidden': prompt if t == 0 else hidden,
                         'prompt_hidden': prompt, 'observable': observable[:, :5] if t == 0 else observable}
                for t in [0, 32]}
    manifest = {'layers': [6], 'pilot_task_ids': ['pilot']}
    result = grouped_predictions(records, features, manifest, config, include_policy=False)
    assert len(result['predictions']) == 80
    for selection in result['selections']:
        assert not set(selection['train_tasks']) & set(selection['test_tasks'])
    zero = [p for p in result['predictions'] if p['checkpoint'] == '0']
    assert within_task_auc([p['passed'] for p in zero], [p['H'] for p in zero], [p['task_id'] for p in zero]) == 0.5
    records[0]['development_only'] = True
    with pytest.raises(ValueError, match='Pilot'):
        grouped_predictions(records, features, manifest, config, include_policy=False)
