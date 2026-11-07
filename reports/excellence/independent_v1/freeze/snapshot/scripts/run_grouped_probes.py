from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd

from slm_steering.research.probes import load_features, run_with_limited_threads
from slm_steering.research.protocol import file_hash, load_config, provenance, read_bank, write_json


def main():
    parser = argparse.ArgumentParser(description='Leakage-controlled nested grouped linear probes and inner-OOF abort selection')
    parser.add_argument('--bank-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/probes/protocol_v0.2')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/protocol_v0.2.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    if config['protocol_version'] != '0.2':
        raise ValueError('The revised inferential pipeline requires a v0.2 bank/config')
    manifest = json.loads((args.bank_dir / 'manifest.json').read_text(encoding='utf-8'))
    records = read_bank(args.bank_dir / 'trajectories.jsonl')
    if manifest['phase'] != 'definitive' or not manifest.get('bank_frozen'):
        raise ValueError('Gate 3: probes require the frozen definitive bank; do not inspect pilot probes')
    if len(records) != manifest['expected_attempts'] or file_hash(args.bank_dir / 'trajectories.jsonl') != manifest['bank_sha256']:
        raise ValueError('Incomplete or modified definitive bank')
    if not manifest.get('causal_invariance', {}).get('passed') or not manifest.get('seed_reproduction_passed'):
        raise ValueError('Gate 1: prefix invariance and seed reproduction are required')
    if file_hash(args.config) != manifest['config_hash']:
        raise ValueError('Analysis config differs from frozen generation config')
    causal_path = args.bank_dir / 'prefix_validation.json'
    if not causal_path.exists():
        raise ValueError('Gate 1: validate the complete checkpoint coverage with validate_bank_prefixes.py')
    causal = json.loads(causal_path.read_text(encoding='utf-8'))
    if (not causal['passed'] or causal['bank_sha256'] != manifest['bank_sha256']
            or causal['config_sha256'] != file_hash(args.config)):
        raise ValueError('Gate 1: complete-bank prefix validation failed or changed')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = {**provenance(args.config), 'bank_manifest_hash': file_hash(args.bank_dir / 'manifest.json'),
                       'bank_sha256': manifest['bank_sha256'], 'status': 'running'}
    write_json(args.output_dir / 'manifest.json', output_manifest)
    try:
        features = load_features(records, args.bank_dir, manifest, config)
        result = run_with_limited_threads(records, features, manifest, config, model_dir=args.output_dir / 'models')
    except Exception as exc:
        output_manifest.update(status='failed', failure={'type': type(exc).__name__, 'message': str(exc)})
        write_json(args.output_dir / 'manifest.json', output_manifest)
        raise
    pd.DataFrame(result['predictions']).to_csv(args.output_dir / 'outer_predictions.csv', index=False)
    write_json(args.output_dir / 'selection_audit.json', result['selections'])
    write_json(args.output_dir / 'policy_replay.json', result['policy_results'])
    output_manifest.update(status='completed', predictions_sha256=file_hash(args.output_dir / 'outer_predictions.csv'))
    write_json(args.output_dir / 'manifest.json', output_manifest)


if __name__ == '__main__':
    main()
