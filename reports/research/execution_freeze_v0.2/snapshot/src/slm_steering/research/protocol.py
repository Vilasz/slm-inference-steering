from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode('utf-8') if isinstance(value, str) else value).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if config['protocol_version'] not in {'0.1', '0.2'}:
        raise ValueError('Unsupported protocol version')
    for key in ('revision', 'tokenizer_revision'):
        if not re.fullmatch(r'[0-9a-f]{40}', config['model'][key]):
            raise ValueError(f'{key} must be an immutable 40-character commit')
    if config['decoding']['top_k'] != 0:
        raise ValueError('Protocol distribution uses temperature and top-p only; top_k must be 0')
    if config['protocol_version'] in {'0.1', '0.2'}:
        fixed = [(config['model']['id'], 'Qwen/Qwen2.5-Coder-1.5B-Instruct'),
                 (config['pilot']['tasks'], 32), (config['pilot']['temperatures'], [0.6, 0.8, 1.0]),
                 (config['pilot']['attempts_per_task'], 10), (config['candidate_bank']['attempts_per_task'], 20),
                 (config['decoding']['max_new_tokens'], 512), (config['decoding']['top_p'], 0.95),
                 (config['statistics']['outer_folds'], 5), (config['activations']['primary_checkpoint'], 32),
                 (config['policy']['minimum_correct_recall'], 0.97), (config['policy']['noninferiority_margin'], 0.03)]
        if any(actual != expected for actual, expected in fixed):
            raise ValueError('A fixed design choice changed; amend/version the protocol before the affected analysis')
    if config['protocol_version'] == '0.2':
        if config.get('schema') != {'label': 'evalplus_base_and_plus_pass', 'risk_set': 'T>t',
                                    'margin': 'temperature_scaled_pre_top_p'}:
            raise ValueError('v0.2 label, risk-set and margin contracts must be explicit')
    return config


def pilot_task_ids(task_ids, config: dict) -> list[str]:
    prefix = config['pilot']['selection_prefix']
    return sorted(task_ids, key=lambda task: (digest(prefix + task), task))[:config['pilot']['tasks']]


def attempt_seed(version: str, benchmark_version: str, task_id: str, attempt_id: int) -> int:
    # 63 bits fit torch.manual_seed and JSON integers without imposing numpy's 32-bit limit.
    return int(digest(canonical([version, benchmark_version, task_id, attempt_id]))[:16], 16) % (2**63)


def block_indices(depth: int, fractions) -> list[int]:
    return sorted({max(0, min(depth - 1, int(np.floor(depth * f + 0.5)) - 1)) for f in fractions})


def token_position(prompt_tokens: int, completion_tokens: int, checkpoint) -> int:
    if checkpoint == 0:
        return prompt_tokens - 1
    offset = completion_tokens if checkpoint == 'final' else int(checkpoint)
    if offset < 1 or offset > completion_tokens:
        raise ValueError('Checkpoint is outside the observed completion risk set')
    return prompt_tokens + offset - 1


def read_bank(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Corrupt append-only bank at line {number}; recover into a new version') from exc
    validate_bank(records)
    return records


def validate_bank(records: list[dict]) -> None:
    seen, seeds = set(), set()
    required = {'protocol_version', 'run_id', 'benchmark_name', 'benchmark_version', 'task_id', 'attempt_id',
                'seed', 'model_id', 'model_revision', 'tokenizer_revision', 'prompt_template_hash',
                'decoding_config', 'prompt', 'raw_completion', 'extracted_code', 'verifier_version', 'passed',
                'error_type', 'generated_tokens', 'finish_reason', 'truncated', 'generation_time_seconds',
                'verification_time_seconds', 'token_ids', 'token_logprobs', 'token_entropies', 'artifact_hashes',
                'prompt_token_ids', 'development_only', 'token_margins'}
    for record in records:
        if required - record.keys():
            raise ValueError(f'Missing provenance: {sorted(required - record.keys())}')
        for key in ('model_revision', 'tokenizer_revision'):
            if not isinstance(record[key], str) or not re.fullmatch(r'[0-9a-f]{40}', record[key]):
                raise ValueError(f'Missing immutable {key} provenance')
        if any(not record[key] for key in ('run_id', 'model_id', 'verifier_version', 'benchmark_version', 'prompt_token_ids')):
            raise ValueError('Empty generation/verification provenance')
        if any(not isinstance(record[key], (float, int)) or not np.isfinite(record[key]) or record[key] < 0
               for key in ('generation_time_seconds', 'verification_time_seconds')):
            raise ValueError('Missing or invalid generation/verification timing provenance')
        if record['seed'] != attempt_seed(record['protocol_version'], record['benchmark_version'], record['task_id'], record['attempt_id']):
            raise ValueError('Attempt seed differs from preassigned identity')
        identity = (record['task_id'], record['attempt_id'])
        if identity in seen or record['seed'] in seeds:
            raise ValueError('Duplicate attempt identity or seed')
        seen.add(identity)
        seeds.add(record['seed'])
        count = record['generated_tokens']
        if not isinstance(record['passed'], bool) or not isinstance(count, int) or count < 1:
            raise ValueError('Invalid correctness label or generated-token count')
        if any(len(record[key]) != count for key in ('token_ids', 'token_logprobs', 'token_entropies', 'token_margins')):
            raise ValueError('Token statistics and actual generation length disagree')
        if any(not np.isfinite(value) or value > 1e-6 for value in record['token_logprobs']) or any(
                not np.isfinite(value) or value < -1e-6 for value in record['token_entropies']):
            raise ValueError('Invalid sampled-distribution log probabilities or entropies')
        for name, value in (('completion', record['raw_completion']), ('code', record['extracted_code']),
                            ('tokens', canonical(record['token_ids']))):
            if record['artifact_hashes'][name] != digest(value):
                raise ValueError(f'Corrupt {name} artifact')
        if record['protocol_version'] == '0.1' and record['truncated'] and record['passed']:
            raise ValueError('Truncation must count as failure under the protocol')
        if record['protocol_version'] == '0.2':
            if record.get('label_definition') != 'evalplus_base_and_plus_pass':
                raise ValueError('Missing functional-label definition')
            statuses = record.get('verifier_status', {})
            if record['passed'] != (statuses.get('base') == 'pass' and statuses.get('plus') == 'pass'):
                raise ValueError('Functional label disagrees with official verifier')
            if record.get('passed_nontruncated') != (record['passed'] and not record['truncated']):
                raise ValueError('Secondary nontruncated endpoint disagrees with functional label')
            if record.get('margin_definition') != 'temperature_scaled_pre_top_p' or any(
                    value is None or not np.isfinite(value) for value in record['token_margins']):
                raise ValueError('v0.2 requires finite pre-top-p margins')


def require_disjoint(train_groups, test_groups, pilot_ids=()) -> None:
    if set(train_groups) & set(test_groups):
        raise ValueError('Task leakage between train and test')
    if set(test_groups) & set(pilot_ids):
        raise ValueError('Pilot task in confirmatory evaluation')


def provenance(config_path: Path) -> dict:
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    packages = {}
    for name in ('torch', 'transformers', 'evalplus', 'numpy', 'scikit-learn', 'statsmodels', 'PyYAML'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'started_at': now(), 'command': [sys.executable, *sys.argv], 'git_commit': git('rev-parse', 'HEAD'),
            'git_dirty': bool(git('status', '--porcelain')), 'config_hash': file_hash(config_path),
            'protocol_hash': file_hash(ROOT / 'RESEARCH_PROTOCOL.md'), 'python': platform.python_version(),
            'platform': platform.platform(), 'packages': packages,
            'source_hashes': {path.relative_to(ROOT).as_posix(): file_hash(path)
                              for path in sorted((ROOT / 'src/slm_steering/research').glob('*.py'))},
            'evalplus_worker_hash': file_hash(ROOT / 'scripts/evalplus_worker.py')}


def require_freeze(config_path: Path, pilot_manifest: dict) -> None:
    version = load_config(config_path)['protocol_version'] if config_path.exists() else 'unavailable'
    execution_record = ROOT / f'reports/research/execution_freeze_v{version}/manifest.json'
    if execution_record.exists():
        from slm_steering.research.freeze import validate_execution_freeze
        validate_execution_freeze(ROOT, config_path)
        if pilot_manifest.get('status') != 'completed' or not pilot_manifest.get('selected_temperature'):
            raise ValueError('Gate 2: full pilot and sampling selection are required')
        freeze_path = ROOT / f'configs/sampling_freeze_v{version}.json'
        frozen = json.loads(freeze_path.read_text(encoding='utf-8'))
        if any(frozen.get(key) != pilot_manifest.get(key) for key in (
                'config_hash', 'selected_temperature', 'prompt_template_hash')):
            raise ValueError('Gate 2: sampling record differs from completed pilot')
        if pilot_manifest['config_hash'] != file_hash(config_path):
            raise ValueError('Gate 2: configuration changed after pilot')
        return
    protocol_path = ROOT / 'RESEARCH_PROTOCOL.md'
    protocol = protocol_path.read_text(encoding='utf-8')
    if not re.search(r'\*\*Status:\*\*\s*frozen\b', protocol):
        raise ValueError('Gate 0: protocol approval and frozen status are required before definitive generation')
    for role in ('Student', 'Advisor'):
        approval = re.search(rf'^\|\s*{role}\s*\|[^|]+\|\s*([^|]+)\|', protocol, re.MULTILINE)
        if not approval or approval.group(1).strip().lower() not in {'approved', 'accepted', 'aprovado', 'aceito'}:
            raise ValueError(f'Gate 0: missing {role} approval record')
    config = load_config(config_path)
    version = config['protocol_version']
    tag = f'research-protocol-v{version}'
    tagged = subprocess.check_output(['git', 'show', f'{tag}:RESEARCH_PROTOCOL.md'], cwd=ROOT)
    if digest(tagged.replace(b'\r\n', b'\n')) != digest(protocol.replace('\r\n', '\n')):
        raise ValueError('Gate 0: working protocol differs from immutable protocol tag')
    if not pilot_manifest.get('selected_temperature') or pilot_manifest.get('status') != 'completed':
        raise ValueError('Gate 2: full pilot and sampling selection are required')
    if pilot_manifest['config_hash'] != file_hash(config_path):
        raise ValueError('Config changed since the pilot; create a new audited version')
    freeze_path = ROOT / f'configs/sampling_freeze_v{version}.json'
    if not freeze_path.exists():
        raise ValueError(f'Gate 2: prepare and commit {freeze_path.name} after the full pilot')
    frozen = json.loads(freeze_path.read_text(encoding='utf-8'))
    if any(frozen.get(key) != pilot_manifest.get(key) for key in ('config_hash', 'selected_temperature', 'prompt_template_hash')):
        raise ValueError('Sampling freeze record differs from completed pilot')
    for relative in ('RESEARCH_PROTOCOL.md', str(config_path.relative_to(ROOT)), str(freeze_path.relative_to(ROOT))):
        subprocess.check_output(['git', 'ls-files', '--error-unmatch', relative], cwd=ROOT, stderr=subprocess.DEVNULL)
        if subprocess.check_output(['git', 'status', '--porcelain', '--', relative], cwd=ROOT, text=True).strip():
            raise ValueError('Protocol, exact sampling config and environment record must be committed before definitive generation')
