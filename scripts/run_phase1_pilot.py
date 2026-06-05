from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roda o piloto recomendado da etapa 1: 5 problemas HumanEval com Best-of-N."
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/phase1_pilot"))
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"humaneval{args.limit}_n{args.n}_seed{args.seed}"
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_baseline.py")),
        "--limit",
        str(args.limit),
        "--n",
        str(args.n),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
        "--output-jsonl",
        str(args.output_dir / f"{stem}.jsonl"),
        "--summary-json",
        str(args.output_dir / f"{stem}_summary.json"),
    ]
    if not args.allow_cpu:
        command.append("--require-cuda")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
