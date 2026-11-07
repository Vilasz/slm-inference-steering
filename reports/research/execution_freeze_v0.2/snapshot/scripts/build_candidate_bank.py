from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

# On Windows this CUDA wheel must load its DLLs before numpy/scipy's OpenMP runtime.
if all(flag not in sys.argv for flag in ('--dry-run', '--help', '-h')):
    import torch  # noqa: F401 -- preload native CUDA DLLs before scientific libraries

from slm_steering.research.bank import build_bank
from slm_steering.research.protocol import load_config


def main():
    parser = argparse.ArgumentParser(description='Protocol v0.2 resumable, append-only Candidate Bank generation')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/protocol_v0.2.yaml')
    parser.add_argument('--phase', choices=['pilot', 'definitive', 'transfer'], default='pilot')
    parser.add_argument('--transfer-bundle', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--pilot-manifest', type=Path)
    parser.add_argument('--time-budget-seconds', type=float)
    parser.add_argument('--max-attempts', type=int, help='New attempts per session; never changes the planned sample')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--wsl-python', help='POSIX Python with EvalPlus installed, e.g. /home/user/.venvs/evalplus/bin/python')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps({'phase': args.phase, 'config': config, 'gate_0_required': args.phase != 'pilot',
                          'planned_pilot_attempts': 32 * 10 * 3, 'planned_definitive_attempts': 132 * 20,
                          'planned_transfer_attempts': config.get('robustness', {}).get('tasks', 0) * 20}, indent=2))
        return
    result = build_bank(args.config, args.output_dir or ROOT / 'runs/candidate_bank' / args.phase, phase=args.phase,
                        time_budget=args.time_budget_seconds or config['resources']['default_session_seconds'],
                        local_files_only=not args.allow_download, wsl_python=args.wsl_python,
                        pilot_manifest_path=args.pilot_manifest, max_attempts=args.max_attempts,
                        transfer_bundle=args.transfer_bundle)
    print(json.dumps({key: result.get(key) for key in ('status', 'completed_attempts', 'expected_attempts',
                     'selected_temperature', 'hardware')}, indent=2))


if __name__ == '__main__':
    main()
