from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd

from slm_steering.research.protocol import file_hash, load_config, write_json
from slm_steering.research.reporting import write_artifacts


def main():
    parser = argparse.ArgumentParser(description='Protocol Q1/Q2 clustered inference and Q3 offline policy artifacts')
    parser.add_argument('--probes-dir', type=Path, default=ROOT / 'runs/probes/protocol_v0.2')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/protocol_v0.2')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/protocol_v0.2.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = json.loads((args.probes_dir / 'manifest.json').read_text(encoding='utf-8'))
    predictions_path = args.probes_dir / 'outer_predictions.csv'
    if manifest['status'] != 'completed' or manifest['predictions_sha256'] != file_hash(predictions_path):
        raise ValueError('Incomplete or modified outer predictions')
    if manifest['config_hash'] != file_hash(args.config):
        raise ValueError('Config differs from frozen analysis manifest')
    predictions = pd.read_csv(predictions_path, dtype={'checkpoint': str})
    policies = pd.DataFrame(json.loads((args.probes_dir / 'policy_replay.json').read_text(encoding='utf-8')))
    online_file = args.probes_dir / 'online_validation.json'
    online_ok = False
    if online_file.exists():
        online = json.loads(online_file.read_text(encoding='utf-8'))
        online_ok = (online.get('passed', False) and online.get('has_actual_abort', False)
                     and online.get('predictions_sha256') == manifest['predictions_sha256']
                     and online.get('bank_sha256') == manifest['bank_sha256'])
    results = write_artifacts(predictions, policies, config, args.output_dir, online_validated=online_ok)
    results['provenance'] = {'config_sha256': file_hash(args.config),
                             'predictions_sha256': manifest['predictions_sha256'],
                             'bank_sha256': manifest['bank_sha256'],
                             'advisor_approval': 'See execution authorization; not inferred from execution',
                             'preregistration_claim': False}
    write_json(args.output_dir / 'results.json', results)
    print(json.dumps({'H2': results['signal']['H2'], 'H3': results['policies'].get('H3')}, indent=2))


if __name__ == '__main__':
    main()
