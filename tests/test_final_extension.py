import itertools

import numpy as np
import pandas as pd
import pytest

from slm_steering.final_extension.decision import finite_budget_policy, simplified_continue
from slm_steering.final_extension.probes import Features, grouped_partition, fit_mlp, mlp_predict
from slm_steering.final_extension.evaluation import paired_auc_contrast
from slm_steering.final_extension.power import simulate
from slm_steering.excellence.economics import simulate as replay
from slm_steering.research.features import observable_features
from slm_steering.research.generation import sampling_statistics


def test_deepseek_official_template_single_bos_and_trailing_newline():
    from transformers import AutoTokenizer
    from slm_steering.final_extension.generation import ExtensionGenerator
    try:
        tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-1.3b-instruct',
                        revision='e063262dac8366fc1f28a4da0ff3c50ea66259ca', local_files_only=True)
    except OSError:
        pytest.skip('Real tokenizer check requires the pinned local DeepSeek tokenizer cache')
    engine = object.__new__(ExtensionGenerator)
    engine.name, engine.tokenizer = 'deepseek', tokenizer
    rendered, ids = engine.encode_prompt('def f(x):\n    """Return x."""')
    assert rendered.endswith('### Response:\n')
    assert ids[0] == tokenizer.bos_token_id and ids[1] != tokenizer.bos_token_id
    expected = tokenizer.apply_chat_template([
        {'role': 'system', 'content': 'You are a careful Python coding assistant. Return only valid Python code, with no markdown or commentary.'},
        {'role': 'user', 'content': 'Complete the following Python function for HumanEval. Return only the code needed to complete the function.\n\n```python\ndef f(x):\n    """Return x."""\n```'}], tokenize=True, add_generation_prompt=True)
    assert ids == expected


def test_pca_scaler_never_fit_held_out_tasks():
    rng = np.random.default_rng(4)
    groups = np.repeat([f't{i}' for i in range(15)], 10)
    data = {'groups': groups, 'hidden': rng.normal(size=(150, 70)), 'observable': rng.normal(size=(150, 10))}
    train, held = grouped_partition(groups)
    first = Features(True).fit(data, np.flatnonzero(train))
    modified = {**data, 'hidden': data['hidden'].copy(), 'observable': data['observable'].copy()}
    modified['hidden'][held] += 100000
    modified['observable'][held] += 100000
    second = Features(True).fit(modified, np.flatnonzero(train))
    assert not set(first.fit_tasks) & set(groups[held])
    np.testing.assert_array_equal(first.pca.components_, second.pca.components_)
    np.testing.assert_array_equal(first.bscale.mean_, second.bscale.mean_)


def test_causal_observables_ignore_future_tokens_and_labels():
    record = {'prompt': 'def f(x):', 'prompt_token_ids': [1, 2], 'signature_token_count': 5,
              'generated_tokens': 40, 'token_logprobs': [-1.] * 40, 'token_entropies': [.4] * 40,
              'token_margins': [2.] * 40, 'passed': True}
    before = observable_features(record, 32)
    record.update(passed=False, generated_tokens=512)
    for key in ('token_logprobs', 'token_entropies', 'token_margins'):
        record[key][32:] = [9999.] * 8
    np.testing.assert_array_equal(before, observable_features(record, 32))


def test_sampling_distribution_and_next_token_quantities():
    torch = pytest.importorskip('torch')
    logits = torch.tensor([[0., 1., 2., -1.]])
    p, lp, entropy, margin = sampling_statistics(logits, 1., 1.)
    expected = torch.softmax(logits, -1)
    torch.testing.assert_close(p, expected)
    torch.testing.assert_close(lp, torch.log(expected))
    torch.testing.assert_close(entropy, -(expected * expected.log()).sum(-1))
    assert margin.item() == 1.
    p, lp, entropy, _ = sampling_statistics(logits, 1., .01)
    assert p.tolist() == [[0., 0., 1., 0.]]
    assert entropy.item() == 0.
    assert torch.isneginf(lp[0, 0])


def test_mlp_seed_and_numpy_prediction():
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(6)
    x = rng.normal(size=(32, 4)).astype(np.float32)
    y = np.tile([0, 1], 16)
    groups = np.repeat(['a', 'b', 'c', 'd'], 8)
    a = fit_mlp(x, y, groups, width=16, seed=12, epochs=5)
    b = fit_mlp(x, y, groups, width=16, seed=12, epochs=5)
    np.testing.assert_array_equal(mlp_predict(a, x), mlp_predict(b, x))
    from slm_steering.final_extension.probes import make_mlp
    model = make_mlp(4, 16)
    model.load_state_dict({k: torch.tensor(v) for k, v in a['state'].items()})
    model.eval()
    with torch.inference_mode():
        expected = model(torch.tensor(x)).flatten().numpy()
    np.testing.assert_allclose(mlp_predict(a, x), expected, atol=1e-7)


def test_paired_auc_ties_homogeneous_and_mismatches():
    frame = pd.DataFrame([{'family': f, 'task_id': t, 'attempt_id': j, 'passed': y, 'probability': p}
                          for f in ('A', 'B') for t in ('mixed', 'homogeneous')
                          for j, y, p in [(0, 1, .5), (1, 0 if t == 'mixed' else 1, .5)]])
    value = paired_auc_contrast(frame, 'A', 'B', replications=100)
    assert value['tasks'] == 1 and value['estimate'] == 0 and value['ci95'] == [0, 0]
    assert value['interpretation'] == 'inconclusive'
    with pytest.raises(ValueError, match='identical candidates'):
        paired_auc_contrast(frame.iloc[:-1], 'A', 'B')
    assert paired_auc_contrast(frame[frame.task_id == 'homogeneous'], 'A', 'B')['tasks'] == 0


def test_replay_strict_risk_set_and_cost():
    stream = [{'task_id': 't', 'attempt_id': j, 'generated_tokens': length, 'passed': passed}
              for j, length, passed in [(0, 32, False), (1, 100, True), (2, 80, True)]]
    scores = {('t', 1): .1, ('t', 2): .9}
    gate = replay(stream, checkpoint=32, threshold=.5, scores=scores, budget=3, kind='gate')
    assert gate['tokens'] == 32 + 32 + 80
    assert gate['false_aborts'] == 1 and gate['resolved'] == 1
    assert gate['active_correct_in_budget'] == 2 and gate['retained_correct_in_budget'] == 1
    fixed = replay(stream, checkpoint=32, budget=3, kind='fixed')
    assert fixed['tokens'] == 212


@pytest.mark.parametrize('p,c,a,penalty,k,b', itertools.product([0., .3, 1.], [1, 3], [1, 2], [0, 5], [0, 1, 2], [0, 2, 5]))
def test_bellman_against_explicit_action_tree(p, c, a, penalty, k, b):
    # Independent enumeration of complete iid attempt policies, in closed form.
    # Starting and then aborting cannot help with known constant p and positive costs.
    values = [(a + c) * sum((1 - p)**j for j in range(n)) + penalty * (1 - p)**n
              for n in range(min(k, b // (a + c)) + 1)]
    assert finite_budget_policy(p, c, a, penalty, k, b)['value'] == pytest.approx(min(values))


def test_simplified_threshold_derived_cost_order():
    for p, v, c in itertools.product([0., .2, 1.], [0., 3., 10.], [0., 1., 5.]):
        assert simplified_continue(p, v, c) == (c + (1 - p) * v <= v + 1e-12)


def test_power_known_effect_and_task_count():
    frame = simulate([-.2, 0, .2], .5, [50, 100], simulations=100, seed=3)
    assert set(frame.effect) == {0., .02, .03, .05, .075}
    assert frame.power_two_sided.between(0, 1).all()
    assert frame.mean_mixed_tasks.between(15, 65).all()


def test_bank_provenance_rejects_seed_and_label_changes():
    from slm_steering.final_extension.bank_audit import audit_record
    from slm_steering.final_extension.preparation import MODELS
    from slm_steering.research.protocol import attempt_seed
    r = {'model': 'deepseek', 'task_id': 'HumanEval/0', 'attempt_id': 1, 'benchmark_version': 'v0.1.10',
         'model_revision': MODELS['deepseek']['revision'], 'tokenizer_revision': MODELS['deepseek']['revision'],
         'generated_tokens': 40, 'token_ids': [1]*40, 'token_logprobs': [-1.]*32, 'token_entropies': [1.]*32,
         'token_margins': [1.]*32, 'passed': False, 'verifier_status': {'base': 'pass', 'plus': 'fail'}}
    r['seed'] = attempt_seed('final-cross-model-v1-deepseek', r['benchmark_version'], r['task_id'], r['attempt_id'])
    assert audit_record(r, [17])
    with pytest.raises(ValueError, match='Seed'):
        audit_record({**r, 'seed': r['seed']+1}, [17])
    with pytest.raises(ValueError, match='Label'):
        audit_record({**r, 'passed': True}, [17])


@pytest.mark.parametrize('complete', [False, True])
def test_manuscript_pending_and_complete_branches_compile(tmp_path, monkeypatch, complete):
    """Synthetic fixtures stay in pytest's temporary directory, never in evidence."""
    pytest.importorskip('typst')
    from slm_steering.final_extension import publication
    from slm_steering.final_extension.common import atomic_json
    from pathlib import Path
    import shutil
    root = tmp_path
    report = root / 'reports/final_extension'
    (root / 'tcc/chapters').mkdir(parents=True)
    (root / 'tcc/build').mkdir()
    report.mkdir(parents=True)
    (root / 'tcc/main.typ').write_text('#include "chapters/06_discussao.typ"', encoding='utf-8')
    (root / 'tcc/chapters/06_discussao.typ').write_text('= Discussão sintética de teste', encoding='utf-8')
    real_report = Path(publication.REPORT)
    shutil.copyfile(real_report / 'power_precision.svg', report / 'power_precision.svg')
    atomic_json(report / 'status.json', {'pilot': None, 'external_results_complete': complete})
    atomic_json(report / 'power_results.json', {'planning_rows': [{'effect': .02, 'positive_detection': .1, 'mean_interval_width': .1}],
                 'paired_delta_sd': .2, 'recommendation_tasks': 80})
    atomic_json(report / 'model_metadata.json', {n: {'files': {'config.json': {'contents': {'num_hidden_layers': 24, 'hidden_size': 2048}}}}
                                                for n in ('qwen', 'deepseek')})
    atomic_json(report / 'task_inventory.json', {'total_tasks': 378, 'exposed_ids': ['a'], 'eligible_ids': ['b']})
    if complete:
        estimate = {'estimate': .01, 'ci95': [-.02, .04], 'tasks': 2, 'interpretation': 'inconclusive'}
        families = ('B_linear', 'HB_linear', 'B_MLP', 'HB_MLP')
        result = {'primary': estimate, 'contrasts': {n: {'linear': estimate, 'MLP': estimate, 'Gamma': estimate} for n in ('deepseek', 'qwen')},
                  'metrics': [{'model': n, 'family': f, 'within_task_auc': .6, 'log_loss': .6, 'brier': .2}
                              for n in ('deepseek', 'qwen') for f in families],
                  'operational': [{'model': n, 'policy': f, 'budget': 20, 'resolved_tasks': 2,
                                   'mean_completion_tokens': 100., 'retention_all_active_correct': 1.,
                                   'false_aborts_started': 0, 'lost_tasks': 0,
                                   'delta_mean_tokens_vs_verifier': estimate, 'delta_coverage_vs_verifier': estimate}
                                  for n in ('deepseek', 'qwen') for f in ('fixed', 'verifier', 'oracle', *families)]}
        atomic_json(report / 'results.json', result)
    monkeypatch.setattr(publication, 'ROOT', root)
    monkeypatch.setattr(publication, 'REPORT', report)
    publication.integrate(compile_pdf=True)
    assert (root / 'tcc/build/manuscrito.pdf').stat().st_size > 1000
    text = (root / 'tcc/chapters/09_robustez_modelos.typ').read_text(encoding='utf-8')
    assert ('Contrastes externos por modelo' in text) == complete
