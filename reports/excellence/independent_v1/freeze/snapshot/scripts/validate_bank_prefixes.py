"""Check every available checkpoint against a stored, outcome-independent trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if all(flag not in sys.argv for flag in ('--help', '-h')):
    import torch  # noqa: F401 -- Windows CUDA DLL preload
import numpy as np

from slm_steering.research.generation import PrefixGenerator
from slm_steering.research.protocol import file_hash, load_config, now, read_bank, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/protocol_v0.2.yaml')
    args = parser.parse_args()
    manifest = json.loads((args.bank_dir / 'manifest.json').read_text(encoding='utf-8'))
    records = read_bank(args.bank_dir / 'trajectories.jsonl')
    if manifest['status'] != 'completed' or len(records) != manifest['expected_attempts']:
        raise ValueError('Validate the completed bank before inference')
    config = load_config(args.config)
    if manifest['config_hash'] != file_hash(args.config):
        raise ValueError('Configuration differs from stored generation')
    needed = max(t for t in config['activations']['checkpoints'] if isinstance(t, int)
                 and any(r['generated_tokens'] >= t for r in records))
    # Eligibility is based on length, never correctness. Prefer the shortest
    # qualifying sequence to bound GPU memory during full-prefix extraction.
    eligible = [r for r in records if r['generated_tokens'] >= needed]
    selected = min(eligible or records, key=lambda r: (r['generated_tokens'], r['task_id'], r['attempt_id']))
    activation = args.bank_dir / selected['activation_path']
    if file_hash(activation) != selected['artifact_hashes']['activations']:
        raise ValueError('Stored activation hash mismatch')
    with np.load(activation, allow_pickle=False) as states:
        sample = {**selected, 'activations': {key: states[key].copy() for key in states.files}}
    generator = PrefixGenerator(config)
    if generator.template_hash != selected['prompt_template_hash']:
        raise ValueError('Prompt template changed')
    result = generator.invariance_test(sample)
    covered = {str(check['checkpoint']) for check in result['checks']}
    required = {str(t) for t in config['activations']['checkpoints']
                if t in (0, 'final') or any(r['generated_tokens'] >= t for r in records)}
    if not required.issubset(covered):
        raise ValueError('A stored checkpoint lacks causal validation coverage')
    result.update(created_at=now(), bank_sha256=file_hash(args.bank_dir / 'trajectories.jsonl'),
                  config_sha256=file_hash(args.config), task_id=selected['task_id'], attempt_id=selected['attempt_id'],
                  selection='Shortest trajectory reaching largest available checkpoint; no outcome filtering',
                  activation_sha256=file_hash(activation), hardware=generator.hardware())
    write_json(args.bank_dir / 'prefix_validation.json', result)
    print(f'All available prefix checkpoints validated: {sorted(covered)}', flush=True)


if __name__ == '__main__':
    main()
