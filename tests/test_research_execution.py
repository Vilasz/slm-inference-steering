from __future__ import annotations

import json
import shutil

import pytest

from slm_steering.research.freeze import prepare_execution_freeze, validate_execution_freeze
from slm_steering.research.protocol import ROOT, file_hash, require_freeze, write_json


def fixture_repository(tmp_path):
    config = tmp_path / 'configs/protocol_v0.2.yaml'
    config.parent.mkdir()
    shutil.copy2(ROOT / 'configs/protocol_v0.2.yaml', config)
    (tmp_path / 'RESEARCH_PROTOCOL.md').write_text('**Status:** draft for advisor approval\n', encoding='utf-8')
    for parent, names in [
        ('src/slm_steering/research', ['generation.py', 'probes.py']),
        ('scripts', ['evalplus_worker.py', 'build_candidate_bank.py', 'prepare_sampling_freeze.py',
                     'run_grouped_probes.py', 'generate_tcc_artifacts.py', 'validate_online_replay.py',
                     'run_frozen_transfer.py', 'validate_bank_prefixes.py', 'run_complete_research.py']),
    ]:
        directory = tmp_path / parent
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).write_text('# fixture source\n', encoding='utf-8')
    return config


def test_execution_authorization_keeps_advisor_pending_and_detects_source_changes(tmp_path):
    config = fixture_repository(tmp_path)
    frozen = prepare_execution_freeze(tmp_path, config, 'Execute the complete experiment')
    assert frozen['advisor_approval'] == 'pending'
    assert frozen['preregistration_claim'] is False
    assert validate_execution_freeze(tmp_path, config) == frozen
    source = tmp_path / 'src/slm_steering/research/generation.py'
    source.write_text('# altered numerical regime\n', encoding='utf-8')
    with pytest.raises(ValueError, match='frozen input changed'):
        validate_execution_freeze(tmp_path, config)


def test_execution_authorization_still_requires_complete_pilot_and_exact_sampling(tmp_path, monkeypatch):
    from slm_steering.research import protocol
    config = fixture_repository(tmp_path)
    prepare_execution_freeze(tmp_path, config, 'Execute the complete experiment')
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    pilot = {'status': 'partial', 'selected_temperature': None}
    with pytest.raises(ValueError, match='full pilot'):
        require_freeze(config, pilot)
    pilot.update(status='completed', selected_temperature=.8, config_hash=file_hash(config), prompt_template_hash='template')
    write_json(tmp_path / 'configs/sampling_freeze_v0.2.json', pilot)
    require_freeze(config, pilot)
    pilot['selected_temperature'] = 1.0
    with pytest.raises(ValueError, match='sampling record differs'):
        require_freeze(config, pilot)


def test_snapshot_tampering_and_retrospective_freeze_are_rejected(tmp_path):
    config = fixture_repository(tmp_path)
    definitive = tmp_path / 'runs/candidate_bank/definitive_v0.2/trajectories.jsonl'
    definitive.parent.mkdir(parents=True)
    definitive.write_text('{"already_observed":true}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='precede definitive outcomes'):
        prepare_execution_freeze(tmp_path, config, 'Execute the experiment')
    definitive.unlink()
    prepare_execution_freeze(tmp_path, config, 'Execute the experiment')
    manifest = tmp_path / 'reports/research/execution_freeze_v0.2/manifest.json'
    record = json.loads(manifest.read_text(encoding='utf-8'))
    record['advisor_approval'] = 'approved'
    write_json(manifest, record)
    with pytest.raises(ValueError, match='impersonate advisor approval'):
        validate_execution_freeze(tmp_path, config)


@pytest.mark.parametrize(('loss_ci', 'loss_finding'), [
    ([-.1, .1], 'O intervalo de perda inclui zero'),
    ([.01, .07], 'O intervalo de perda permanece acima de zero'),
    ([-.07, -.01], 'O intervalo de perda permanece abaixo de zero'),
])
def test_manuscript_renderer_preserves_negative_and_unidentified_findings(tmp_path, monkeypatch, loss_ci, loss_finding):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    from build_manuscript import interval, number, table
    from render_main_manuscript import render
    root = tmp_path
    (root / 'tcc').mkdir()
    (root / 'tcc/main.typ').write_text('versão empírica exploratória', encoding='utf-8')
    result = {'checkpoint': '32', 'roc_auc': .5, 'within_task_auc': None,
              'within_task_auc_ci95': None, 'log_loss': .693}
    evidence = {'main': {'signal': {
        'H2': {'supported': False, 'delta_hb_minus_b': None,
               'delta_log_loss_hb_minus_b': {'estimate': sum(loss_ci) / 2, 'ci95': loss_ci},
               'probabilistic_gain_uncertain': loss_ci[1] >= 0},
        'metrics': [{**result, 'predictor': name} for name in ('B', 'H', 'HB')],
        'risk_sets': [{'checkpoint': '32', 'trajectories': 2000, 'tasks': 132, 'mixed_tasks': 0}]},
        'policies': {'H3': {'operational_benefit_claim_supported': False}, 'summaries': []}}}
    render(root, evidence, number, interval, table)
    content = (root / 'tcc/generated/resultados.typ').read_text(encoding='utf-8')
    assert 'não satisfaz' in content
    assert 'não identificável' in content
    assert 'não foram satisfeitos' in content
    assert loss_finding in content
    abstract = (root / 'tcc/generated/resumo.typ').read_text(encoding='utf-8')
    assert 'transferência congelada para MBPP+ permanece pendente' in abstract
