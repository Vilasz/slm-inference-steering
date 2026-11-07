from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slm_steering.experiments.runner import run_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa uma pipeline experimental configurada.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment_config(
        args.config,
        root=ROOT,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

