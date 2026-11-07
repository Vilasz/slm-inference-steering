"""Auditable execution authorization, distinct from advisor approval."""
from __future__ import annotations

import json
from pathlib import Path

from slm_steering.research.protocol import file_hash, load_config, now, write_json


def validate_execution_freeze(root: Path, config_path: Path) -> dict:
    version = load_config(config_path)['protocol_version']
    directory = root / f'reports/research/execution_freeze_v{version}'
    record = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if (record.get('authorization_source') != 'explicit_user_request'
            or not record.get('authorization_text')
            or record.get('advisor_approval') != 'pending'
            or record.get('preregistration_claim') is not False):
        raise ValueError('Gate 0: execution authorization must not impersonate advisor approval')
    required = {'RESEARCH_PROTOCOL.md', config_path.relative_to(root).as_posix(),
                'scripts/evalplus_worker.py'}
    required.update(p.relative_to(root).as_posix() for p in (root / 'src/slm_steering/research').glob('*.py'))
    if not required.issubset(record['inputs']):
        raise ValueError('Gate 0: incomplete execution snapshot')
    for relative, expected in record['inputs'].items():
        for candidate in (root / relative, directory / 'snapshot' / relative):
            if file_hash(candidate) != expected:
                raise ValueError(f'Gate 0: frozen input changed: {relative}')
    if record['config_hash'] != file_hash(config_path):
        raise ValueError('Gate 0: configuration differs from execution snapshot')
    return record


def prepare_execution_freeze(root: Path, config_path: Path, authorization_text: str) -> dict:
    version = load_config(config_path)['protocol_version']
    directory = root / f'reports/research/execution_freeze_v{version}'
    if (directory / 'manifest.json').exists():
        return validate_execution_freeze(root, config_path)
    if not authorization_text.strip():
        raise ValueError('An explicit, recorded user request is required')
    definitive = root / f'runs/candidate_bank/definitive_v{version}/trajectories.jsonl'
    if definitive.exists() and definitive.stat().st_size:
        raise ValueError('Execution freeze must precede definitive outcomes')
    paths = [root / 'RESEARCH_PROTOCOL.md', config_path, root / 'scripts/evalplus_worker.py']
    paths.extend(root / name for name in ('requirements.txt', 'requirements-evalplus.txt', 'uv.lock',
                                         'docs/execution_amendment_v0.2.md') if (root / name).exists())
    paths.extend(sorted((root / 'src/slm_steering/research').glob('*.py')))
    paths.extend(root / 'scripts' / name for name in (
        'build_candidate_bank.py', 'prepare_sampling_freeze.py', 'run_grouped_probes.py',
        'generate_tcc_artifacts.py', 'validate_online_replay.py', 'run_frozen_transfer.py',
        'validate_bank_prefixes.py', 'run_complete_research.py'))
    inputs = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        target = directory / 'snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(path.read_bytes())
        inputs[relative] = file_hash(path)
    record = {'created_at': now(), 'protocol_version': version,
              'authorization_source': 'explicit_user_request', 'authorization_text': authorization_text,
              'advisor_approval': 'pending', 'preregistration_claim': False,
              'evidence_scope': 'Fixed-design prospective execution; legacy exposure disclosed in protocol',
              'config_hash': file_hash(config_path), 'inputs': inputs,
              'online_validation_tasks': 12, 'online_task_selection': 'Lexicographic task IDs, no outcome filtering'}
    write_json(directory / 'manifest.json', record)
    return validate_execution_freeze(root, config_path)
