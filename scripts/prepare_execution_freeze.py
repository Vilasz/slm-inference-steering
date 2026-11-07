from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slm_steering.research.freeze import prepare_execution_freeze


def main():
    parser = argparse.ArgumentParser(description='Snapshot the design under explicit execution authorization; no advisor approval')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/protocol_v0.2.yaml')
    parser.add_argument('--authorization-text', required=True)
    args = parser.parse_args()
    record = prepare_execution_freeze(ROOT, args.config.resolve(), args.authorization_text)
    print(json.dumps({'created_at': record['created_at'], 'advisor_approval': record['advisor_approval'],
                      'preregistration_claim': record['preregistration_claim'], 'inputs': len(record['inputs'])}, indent=2))


if __name__ == '__main__':
    main()
