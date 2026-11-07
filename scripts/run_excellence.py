"""Versioned research improvements; independent confirmatory data require explicit approval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'models', 'transfer', 'economics', 'benchmark', 'hardware_metrics', 'geometry', 'prefixes', 'report'])
    args = parser.parse_args()
    if args.stage == 'benchmark':
        # On this Windows runtime torch's CUDA DLLs must initialize before scipy/sklearn.
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA benchmark requires a visible GPU; no silent CPU fallback')
    from slm_steering.excellence import audit
    if args.stage == 'audit':
        result = audit.run(ROOT)
        print(json.dumps(result, indent=2))
    else:
        import importlib
        module = importlib.import_module('slm_steering.excellence.' + args.stage)
        module.run(ROOT)


if __name__ == '__main__':
    main()
