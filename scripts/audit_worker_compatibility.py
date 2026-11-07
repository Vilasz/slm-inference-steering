"""Explicitly audit the MBPP hash-serialization patch against a partial pilot.

Preserves original provenance and permits the new worker only after every
existing HumanEval+ label and dataset hash has been reproduced. This is a
technical compatibility event, not a scientific endpoint amendment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slm_steering.research.evaluation import EvalPlusClient
from slm_steering.research.protocol import file_hash, now, read_bank, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank-dir', type=Path, default=ROOT / 'runs/candidate_bank/pilot_v0.2')
    parser.add_argument('--wsl-python', required=True)
    args = parser.parse_args()
    manifest_path = args.bank_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['phase'] != 'pilot' or manifest.get('bank_frozen') or (args.bank_dir / '.writer.lock').exists():
        raise ValueError('Compatibility audits require a stopped, unfrozen development bank')
    records = read_bank(args.bank_dir / 'trajectories.jsonl')
    checked = []
    with EvalPlusClient('humaneval', wsl_python=args.wsl_python) as client:
        if client.dataset['metadata']['dataset_sha256'] != manifest['benchmark']['dataset_sha256']:
            raise ValueError('HumanEval dataset serialization changed')
        if client.dataset['metadata']['evalplus_version'] != manifest['benchmark']['evalplus_version']:
            raise ValueError('Official checker version changed')
        for record in records:
            result = client.verify(record['task_id'], record['extracted_code'])
            if result['passed'] != record['passed'] or result['verifier_status'] != record['verifier_status']:
                raise ValueError('Observed HumanEval functional label changed; do not resume this bank')
            checked.append({'task_id': record['task_id'], 'attempt_id': record['attempt_id'], 'matched': True})
    new_hash = file_hash(ROOT / 'scripts/evalplus_worker.py')
    event = {'at': now(), 'old_sha256': manifest.get('compatible_worker_updates', [{}])[-1].get(
                    'new_sha256', manifest['evalplus_worker_hash']), 'new_sha256': new_hash,
             'reason': 'Typed complex/set serialization for MBPP dataset hashing; HumanEval hash and all stored labels reproduced',
             'checked_trajectories': len(checked), 'dataset_sha256': manifest['benchmark']['dataset_sha256'],
             'bank_sha256_unchanged': file_hash(args.bank_dir / 'trajectories.jsonl'),
             'checks': checked, 'passed': True}
    report = ROOT / 'reports/research/worker_compatibility_v0.2.json'
    write_json(report, event)
    manifest.setdefault('compatible_worker_updates', []).append({k: v for k, v in event.items() if k != 'checks'})
    write_json(manifest_path, manifest)
    print(f'{len(checked)} labels and dataset hash reproduced; original provenance preserved')


if __name__ == '__main__':
    main()
