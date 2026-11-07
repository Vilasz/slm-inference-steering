from __future__ import annotations

from copy import deepcopy
import subprocess
import sys

import numpy as np
import pytest

from slm_steering.research.policy import replay, select_policy
from slm_steering.research.probes import choose_model, calibrated_scores
from slm_steering.research.protocol import ROOT, attempt_seed, load_config, validate_bank
from slm_steering.research.statistics import fold_association_test
from test_research_protocol import record


def test_completed_at_checkpoint_is_verified_before_abort():
    records = [record(attempt=0, passed=True, length=32), record(attempt=1, length=80)]
    row = replay(records, {}, checkpoint=32, threshold=0.99, budget=2)[0]
    assert row['resolved'] == 1 and row['completion_tokens'] == 32
    assert row['correct_false_aborts'] == 0 and row['attempts_started'] == 1
    assert not row['decisions'][0]['active_at_checkpoint']


def test_cost_search_can_reject_largest_recall_admissible_threshold():
    # One of 100 positives may be aborted at a 97% target, but losing it
    # initiates a long failure. A recall-only maximum is not cost-optimal.
    records, scores = [], {}
    for task in range(100):
        for attempt in range(2):
            item = record(f't{task}', attempt, passed=attempt == 0, length=40 if attempt == 0 else 500)
            records.append(item)
            scores[(item['task_id'], attempt)] = 0.1 if task == 0 and attempt == 0 else 0.9
    chosen = select_policy(records, [{'checkpoint': 32, 'probabilities': scores}], minimum_recall=0.97, budget=2)
    assert chosen['no_abort']
    assert chosen['mean_tokens'] == 40
    assert any(option['mean_tokens'] > 40 for option in chosen['candidates'])


def test_policy_no_active_positives_uses_no_abort_and_reports_undefined_recall():
    records = [record(length=40), record(attempt=1, length=32, passed=True)]
    chosen = select_policy(records, [{'checkpoint': 32, 'probabilities': {('task', 0): 0.1}}],
                           minimum_recall=0.97, budget=2)
    assert chosen['no_abort'] and chosen['validation_correct_recall'] is None


def test_functional_label_allows_truncated_program_that_passes():
    item = record(passed=True)
    item.update(protocol_version='0.2', truncated=True, finish_reason='length',
                seed=attempt_seed('0.2', 'v1', 'task', 0), verifier_status={'base': 'pass', 'plus': 'pass'},
                label_definition='evalplus_base_and_plus_pass', passed_nontruncated=False,
                margin_definition='temperature_scaled_pre_top_p')
    validate_bank([item])
    item['passed'] = False
    with pytest.raises(ValueError, match='Functional label'):
        validate_bank([item])


def test_fold_permutation_checks_task_isolation_and_handles_empty_folds():
    result = fold_association_test([0, 1], [0.1, 0.9], ['a', 'a'], [0, 0],
                                   replications=19, seed=7, n_folds=5)
    assert all(row['p_value'] == 1 for row in result['folds'][1:])
    assert result['p_value'] == min(1, 5 * min(row['p_value'] for row in result['folds']))
    with pytest.raises(ValueError, match='multiple outer folds'):
        fold_association_test([0, 1], [0.1, 0.9], ['a', 'a'], [0, 1], replications=19, seed=7, n_folds=5)


def test_selection_crossfit_excludes_every_scored_task():
    config = load_config(ROOT / 'configs/protocol_v0.2.yaml')
    config['statistics'].update(regularization_c=[0.1], inner_folds=2)
    groups = np.repeat(np.arange(8), 4)
    y = np.tile([0, 1, 0, 1], 8)
    x = np.random.default_rng(4).normal(size=(32, 3))
    fitted = choose_model([(6, x)], y, groups, config)
    for part in fitted['calibration_audit']:
        assert not set(part['fit_tasks']) & set(part['score_tasks'])
    # Labels of an external set are not arguments to scoring or source fitting.
    target = np.random.default_rng(5).normal(size=(4, 3))
    first = calibrated_scores(fitted, target)
    np.testing.assert_array_equal(first, calibrated_scores(fitted, target))


def test_changing_outer_test_labels_cannot_change_its_models_or_policy():
    from slm_steering.research.features import observable_features
    from slm_steering.research.probes import grouped_predictions
    from sklearn.model_selection import GroupKFold
    config = load_config(ROOT / 'configs/protocol_v0.2.yaml')
    config['statistics'].update(regularization_c=[0.1], inner_folds=2, outer_folds=2)
    config['candidate_bank'].update(attempts_per_task=2, budgets=[1, 2])
    config['policy']['checkpoints'] = [32]
    records = [record(f't{task}', attempt, passed=attempt == 0) for task in range(8) for attempt in range(2)]
    groups = np.array([r['task_id'] for r in records])
    rng = np.random.default_rng(19)
    features = {'32': {'indices': np.arange(16), 'hidden': rng.normal(size=(16, 1, 2)),
                       'prompt_hidden': np.repeat(rng.normal(size=(8, 1, 2)), 2, axis=0),
                       'observable': np.array([observable_features(r, 32) for r in records])}}
    manifest = {'layers': [6], 'pilot_task_ids': []}
    original = grouped_predictions(records, features, manifest, config)
    _, test = next(GroupKFold(2).split(np.zeros(16), groups=groups))
    changed = deepcopy(records)
    for index in test:
        changed[index]['passed'] = not changed[index]['passed']
    perturbed = grouped_predictions(changed, features, manifest, config)
    assert ([s for s in original['selections'] if s['fold'] == 0] ==
            [s for s in perturbed['selections'] if s['fold'] == 0])
    for before, after in zip([p for p in original['predictions'] if p['fold'] == 0],
                             [p for p in perturbed['predictions'] if p['fold'] == 0]):
        for family in ('B', 'H', 'HB'):
            assert before[family] == after[family]


def test_frozen_transfer_does_not_use_target_labels(monkeypatch):
    from slm_steering.research import transfer
    config = load_config(ROOT / 'configs/protocol_v0.2.yaml')
    config['statistics'].update(regularization_c=[0.1], inner_folds=2)
    y, groups = np.tile([0, 1], 8), np.repeat(np.arange(8), 2)
    x = np.random.default_rng(8).normal(size=(16, 1))
    fitted = choose_model([(6, x)], y, groups, config)
    combined = choose_model([(6, np.column_stack((x, x)))], y, groups, config)
    bundle = {'layers': [6], 'models': {('B', 32): fitted, ('H', 32): fitted, ('HB', 32): combined}}
    records = [record('target', i, passed=bool(i % 2)) for i in range(4)]
    features = {'32': {'indices': np.arange(4), 'observable': x[:4], 'hidden': x[:4, None]}}
    def forbidden(*args, **kwargs):
        raise AssertionError('Target transfer attempted fitting')
    monkeypatch.setattr(transfer, 'choose_model', forbidden)
    _, first = transfer.score_frozen(bundle, records, features)
    changed = deepcopy(records)
    for item in changed:
        item['passed'] = not item['passed']
    _, second = transfer.score_frozen(bundle, changed, features)
    assert first == second
    empty = {'32': {'indices': np.array([], dtype=int)}}
    rows, empty_scores = transfer.score_frozen(bundle, records, empty)
    assert rows == [] and all(values == {} for values in empty_scores.values())


def test_real_generation_order_eos_limit_and_pre_top_p_margin():
    code = '''import torch
import numpy as np
from types import SimpleNamespace
from slm_steering.research import generation as g
class Tokenizer:
    def encode(self, text): return [1, 2]
    def decode(self, tokens, **kwargs): return 'program'
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Identity()
    def forward(self, input_ids, past_key_values, **kwargs):
        self.block(torch.ones((1, input_ids.shape[-1], 2)))
        return SimpleNamespace(logits=torch.tensor([[[0., 10.]]]), past_key_values=past_key_values)
g.render_code_prompt = lambda tokenizer, prompt: prompt
g.resolve_transformer_layers = lambda model: [model.block]
gen = g.PrefixGenerator.__new__(g.PrefixGenerator)
gen.torch, gen.device, gen.layers = torch, 'cpu', [0]
gen.model, gen.tokenizer = Model(), Tokenizer()
gen.config = {'decoding': {'max_new_tokens': 2, 'top_p': .5}, 'activations': {'checkpoints': [0, 1, 2]}}
gen.eos_ids = [1]
def forbidden(*args): raise AssertionError('Gate ran on a finished candidate')
sample = gen.generate('def f():', 7, .8, abort=forbidden)
assert sample['finish_reason'] == 'eos' and sample['generated_tokens'] == 1
assert sample['token_margins'] == [12.5]
gen.eos_ids = []
gen.config['decoding']['max_new_tokens'] = 1
sample = gen.generate('def f():', 7, .8, abort=forbidden)
assert sample['finish_reason'] == 'length'
gen.config['decoding']['max_new_tokens'] = 2
sample = gen.generate('def f():', 7, .8, abort=lambda *args: True)
assert sample['finish_reason'] == 'abort' and sample['generated_tokens'] == 1
'''
    completed = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=90,
                               env={**__import__('os').environ, 'PYTHONPATH': str(ROOT / 'src')})
    assert completed.returncode == 0, completed.stderr


def test_mbpp_hash_handles_complex_sets_without_changing_json_only_data():
    import json
    from scripts.evalplus_worker import dataset_json_default
    plain = {'task': {'input': [1, None, 'text', [2.3]], 'prompt': 'function'}}
    assert json.dumps(plain, sort_keys=True) == json.dumps(plain, sort_keys=True, default=dataset_json_default)
    first = {'input': [{3, 1, 2}, complex(1, -2)]}
    second = {'input': [{2, 3, 1}, complex(1, -2)]}
    encoded = json.dumps(first, sort_keys=True, default=dataset_json_default)
    assert encoded == json.dumps(second, sort_keys=True, default=dataset_json_default)
    assert json.loads(encoded)['input'][1] == {'__evalplus_type__': 'complex', 'real': 1.0, 'imag': -2.0}
