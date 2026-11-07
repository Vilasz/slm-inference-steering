from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slm_steering.research.bank import select_temperature
from slm_steering.research.protocol import file_hash, load_config, now, read_bank, write_json


def main():
    parser = argparse.ArgumentParser(description='Prepare the Gate 2 config/environment record after the FULL pilot')
    parser.add_argument('--pilot-dir', type=Path, default=ROOT / 'runs/candidate_bank/pilot_v0.2')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/protocol_v0.2.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'configs/sampling_freeze_v0.2.json')
    args = parser.parse_args()
    manifest = json.loads((args.pilot_dir / 'manifest.json').read_text(encoding='utf-8'))
    config = load_config(args.config)
    records = read_bank(args.pilot_dir / 'trajectories.jsonl')
    if manifest['config_hash'] != file_hash(args.config):
        raise ValueError('Config changed since the pilot')
    selection = select_temperature(records, config, manifest['pilot_task_ids'])
    if manifest['status'] != 'completed' or not selection['complete'] or selection['selected_temperature'] is None:
        raise ValueError('Gate 2: incomplete pilot or no eligible sampling regime')
    frozen = {'created_at': now(), 'protocol_version': config['protocol_version'],
              'config_hash': file_hash(args.config), 'selected_temperature': selection['selected_temperature'],
              'prompt_template_hash': manifest['prompt_template_hash'], 'model': config['model'],
              'benchmark': manifest['benchmark'], 'packages': manifest['packages'], 'hardware': manifest['hardware'],
              'pilot_bank_sha256': file_hash(args.pilot_dir / 'trajectories.jsonl'),
              'pilot_manifest_sha256': file_hash(args.pilot_dir / 'manifest.json'),
              'selection': selection, 'approval_status': 'Recorded in RESEARCH_PROTOCOL.md; not granted by this command'}
    write_json(args.output, frozen)
    print(f'Sampling record ready for versioning: {args.output}')


if __name__ == '__main__':
    main()
