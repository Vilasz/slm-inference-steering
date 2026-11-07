"""Plan or resume the reviewed independent extension; never infer scientific approval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['technical', 'main'], default='technical')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--evaluate', action='store_true')
    parser.add_argument('--wsl-python', default='/home/joaof/.venvs/slm-evalplus-v031/bin/python')
    parser.add_argument('--time-budget-seconds', type=int, default=600)
    args = parser.parse_args()
    from slm_steering.excellence.approvals import approved_design
    design = approved_design(ROOT, require_approval=not args.dry_run)
    # Initialize torch before scientific DLLs on Windows, only after the approval gate.
    if not args.dry_run and not args.evaluate:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Approved collection requires CUDA; no silent CPU fallback')
    if args.dry_run:
        ids = design['technical_pilot_task_ids' if args.phase == 'technical' else 'main_task_ids']
        print(json.dumps({'phase': args.phase, 'tasks': len(ids), 'attempts': len(ids) * 20,
                          'collection_started': False, 'approval_status': 'pending recorded student approval and advisor review'}, indent=2))
    elif args.evaluate:
        from slm_steering.excellence.independent import evaluate
        print(json.dumps(evaluate(ROOT), indent=2))
    else:
        sys.path.insert(0, str(ROOT / 'scripts/research_runtime'))
        from slm_windows_atomic_io import install
        install()
        from slm_steering.excellence.independent import collect
        collect(ROOT, args.phase, wsl_python=args.wsl_python, time_budget=args.time_budget_seconds)


if __name__ == '__main__':
    main()
