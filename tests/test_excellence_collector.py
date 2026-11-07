from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from slm_steering.excellence import independent


def environment(tmp_path, monkeypatch, *, fail=False):
    from slm_steering.excellence import benchmark
    from slm_steering.research import evaluation, generation, protocol
    design = {'technical_pilot_task_ids': ['Mbpp/999'], 'main_task_ids': ['Mbpp/1000'],
              'approved_hardware_device': 'fixtureGPU', 'benchmark_sha256': 'dataset'}
    for file in ['docs/protocols/EXCELLENCE_EXTENSION_v1.md', 'reports/excellence/proposal_design.json',
                 'src/slm_steering/research/generation.py', 'src/slm_steering/excellence/benchmark.py',
                 'runs/transfer/protocol_v0.2/models.joblib', 'configs/protocol_v0.2.yaml']:
        p = tmp_path / file
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}', encoding='utf-8')
    (tmp_path / 'runs/transfer/protocol_v0.2/models.json').write_text(json.dumps({'selections': [{'family': 'HB', 'checkpoint': 8, 'layer': 13}]}))
    monkeypatch.setattr(independent, 'approved_design', lambda root: design)
    monkeypatch.setattr(independent, 'assert_preserved', lambda root: None)
    monkeypatch.setattr(protocol, 'load_config', lambda path: {'model': {'id': 'fixture', 'revision': 'a' * 40, 'tokenizer_revision': 'a' * 40}, 'decoding': {'top_p': .95}})
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(__version__='fixture'))
    sample = {'raw_completion': 'def f(x):\n    return x\n', 'prompt_token_ids': [1, 2], 'token_ids': list(range(9)),
              'token_logprobs': [-.5] * 9, 'token_entropies': [1.] * 9, 'token_margins': [2.] * 9,
              'generated_tokens': 9, 'finish_reason': 'eos', 'truncated': False, 'generation_time_seconds': .1,
              'signature_token_count': 4, 'activations': {k: np.zeros((4, 2), dtype=np.float16) for k in ['0', '8', 'final']}}
    calls = []
    class Engine:
        layers = [6, 13, 20, 27]
        template_hash = 'template'
        def hardware(self):
            return {'device': 'fixtureGPU'}
        def generate(self, *args):
            return copy.deepcopy(sample)
        def invariance_test(self, sample):
            return {'passed': True}
    monkeypatch.setattr(generation, 'PrefixGenerator', lambda config: Engine())
    monkeypatch.setattr(benchmark, 'selective_generate', lambda *args: copy.deepcopy(sample))
    class Client:
        dataset = {'metadata': {'dataset_sha256': 'dataset', 'version': 'v1', 'evalplus_version': '0.3.1'},
                   'tasks': [{'task_id': 'Mbpp/999', 'prompt': 'def f(x):\n    ', 'entry_point': 'f'}]}
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def request(self, value):
            return {'passed': True}
        def verify(self, task, code):
            if task == 'Mbpp/2':
                return {'passed': False}
            calls.append(task)
            if fail:
                raise RuntimeError('fixture worker crash')
            return {'passed': True, 'error_type': None, 'verification_time_seconds': .01}
    monkeypatch.setattr(evaluation, 'EvalPlusClient', Client)
    return calls


def test_collector_is_append_only_idempotent_and_stores_only_selected_states(tmp_path, monkeypatch):
    calls = environment(tmp_path, monkeypatch)
    first = independent.collect(tmp_path, 'technical', wsl_python='fixture', time_budget=60)
    bank = tmp_path / 'runs/excellence/independent_v1/technical/trajectories.jsonl'
    original = bank.read_bytes()
    assert first['status'] == 'completed' and first['completed_attempts'] == 20
    second = independent.collect(tmp_path, 'technical', wsl_python='fixture', time_budget=60)
    assert second['status'] == 'completed' and bank.read_bytes() == original and len(calls) == 20
    record = json.loads(bank.read_text().splitlines()[0])
    with np.load(bank.parent / record['activation_path']) as states:
        assert states['8'].shape == (1, 2) and states['layers'].tolist() == [13]


def test_worker_crash_never_creates_a_negative_candidate(tmp_path, monkeypatch):
    environment(tmp_path, monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match='worker crash'):
        independent.collect(tmp_path, 'technical', wsl_python='fixture', time_budget=60)
    directory = tmp_path / 'runs/excellence/independent_v1/technical'
    assert not (directory / 'trajectories.jsonl').exists()
    assert not (directory / '.writer.lock').exists()
    assert json.loads((directory / 'failure.json').read_text())['type'] == 'RuntimeError'
