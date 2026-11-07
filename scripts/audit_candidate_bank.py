from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from slm_steering.research.protocol import file_hash, now, read_bank, write_json


def audit_bank(bank_dir: Path) -> dict:
    manifest = json.loads((bank_dir / 'manifest.json').read_text(encoding='utf-8'))
    records = read_bank(bank_dir / 'trajectories.jsonl')
    identities = set()
    for record in records:
        if record['development_only'] != (manifest['phase'] == 'pilot'):
            raise ValueError('Development-only flag disagrees with bank phase')
        if manifest['phase'] == 'definitive' and record['task_id'] in manifest['pilot_task_ids']:
            raise ValueError('Pilot task in definitive bank')
        if record['benchmark_hash'] != manifest['benchmark']['dataset_sha256']:
            raise ValueError('Benchmark hash disagrees with manifest')
        if record['model_revision'] != manifest['config']['model']['revision'] or record['prompt_template_hash'] != manifest['prompt_template_hash']:
            raise ValueError('Model or prompt-template provenance disagrees with manifest')
        path = bank_dir / record['activation_path']
        if file_hash(path) != record['artifact_hashes']['activations']:
            raise ValueError('Activation hash mismatch')
        with np.load(path, allow_pickle=False) as states:
            for checkpoint in manifest['config']['activations']['checkpoints']:
                eligible = checkpoint in (0, 'final') or int(checkpoint) <= record['generated_tokens']
                if eligible != (str(checkpoint) in states.files):
                    raise ValueError('Checkpoint risk-set/artifact mismatch')
                if eligible and not np.isfinite(states[str(checkpoint)]).all():
                    raise ValueError('Non-finite activation artifact')
        identities.add((record['task_id'], record['attempt_id']))
    expected = manifest['expected_attempts']
    complete = len(records) == expected
    if manifest.get('bank_frozen') and (not complete or file_hash(bank_dir / 'trajectories.jsonl') != manifest['bank_sha256']):
        raise ValueError('Frozen bank completeness/hash mismatch')
    return {'created_at': now(), 'passed': True, 'complete': complete, 'attempts': len(records),
            'expected_attempts': expected, 'unique_identities': len(identities), 'phase': manifest['phase'],
            'tasks': len({r['task_id'] for r in records}),
            'generated_completion_tokens': sum(r['generated_tokens'] for r in records),
            'bank_sha256': file_hash(bank_dir / 'trajectories.jsonl') if records else None,
            'confirmatory_ready': complete and manifest.get('bank_frozen', False)}


def main():
    parser = argparse.ArgumentParser(description='Validate bank provenance, token arrays, hashes and checkpoint risk sets')
    parser.add_argument('--bank-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit_bank(args.bank_dir)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
