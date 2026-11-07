from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slm_steering.research.transfer import evaluate_transfer, prepare_bundle


def main():
    parser = argparse.ArgumentParser(description='Fit on HumanEval only, then apply a frozen bundle to MBPP+')
    parser.add_argument('operation', choices=['prepare', 'evaluate'])
    parser.add_argument('--bank-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/protocol_v0.2.yaml')
    parser.add_argument('--probes-dir', type=Path, default=ROOT / 'runs/probes/protocol_v0.2')
    parser.add_argument('--bundle', type=Path, default=ROOT / 'runs/transfer/protocol_v0.2/models.joblib')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/transfer_v0.2')
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare_bundle(args.bank_dir, args.probes_dir, args.config, args.bundle)
    else:
        evaluate_transfer(args.bank_dir, args.bundle, args.config, args.output_dir)
    print(f'Frozen transfer {args.operation}: completed')


if __name__ == '__main__':
    main()
