from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slm_steering.metrics import summarize
from slm_steering.reporting import read_jsonl, write_run_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera relatorio Markdown e CSVs a partir de um JSONL de baseline."
    )
    parser.add_argument("jsonl", type=Path, help="Arquivo JSONL gerado por run_baseline.py.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Resumo JSON existente para preservar config/CUDA/artifacts.",
    )
    parser.add_argument(
        "--requested-n",
        type=int,
        default=None,
        help="N usado no experimento, caso o resumo precise ser recalculado.",
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=None,
        help="Prefixo dos artefatos. Padrao: mesmo nome do JSONL sem extensao.",
    )
    parser.add_argument(
        "--update-summary-json",
        action="store_true",
        help="Reescreve o summary JSON com as metricas recalculadas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.jsonl)
    existing_summary = {}
    if args.summary_json and args.summary_json.exists():
        existing_summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    requested_n = (
        args.requested_n
        or existing_summary.get("config", {}).get("n")
        or existing_summary.get("requested_n")
        or max(len(record["attempts"]) for record in records)
    )
    summary = summarize(records, requested_n=requested_n)
    for key in ["config", "cuda"]:
        if key in existing_summary:
            summary[key] = existing_summary[key]
    prefix = args.prefix or args.jsonl.with_suffix("")
    paths = write_run_artifacts(records, summary, prefix)
    summary["artifacts"] = paths
    if args.summary_json and args.update_summary_json:
        args.summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(paths, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
