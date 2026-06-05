from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = [
    ("num_tasks", "Tarefas"),
    ("requested_n", "N solicitado"),
    ("adaptive_sampling_used", "Adaptativo"),
    ("total_attempts", "Tentativas"),
    ("solved_tasks", "Resolvidas"),
    ("strict_pass_at_1", "pass@1 estrito"),
    ("observed_best_of_n", "Best-of-N observado"),
    ("total_generated_tokens", "Tokens gerados"),
    ("total_tokens_until_success_or_budget", "Tokens ate sucesso/orcamento"),
    ("mean_attempts_until_success_or_budget", "Amostras medias ate sucesso/orcamento"),
    ("generated_tokens_per_solved_task", "Tokens por resolvida"),
    ("effective_tokens_per_solved_task", "Tokens efetivos por resolvida"),
    ("total_generation_seconds", "Segundos de geracao"),
    ("mean_generation_tokens_per_second", "Tokens/s medio"),
    ("token_savings_if_oracle_early_stop", "Economia token early-stop ideal"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara resumos JSON de runs baseline/Best-of-N/early-stop/steering."
    )
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--output-md", type=Path, default=Path("runs/comparison.md"))
    parser.add_argument("--output-csv", type=Path, default=Path("runs/comparison.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    labels = args.labels or [path.stem.replace("_summary", "") for path in args.summaries]
    if len(labels) != len(summaries):
        raise ValueError("--labels deve ter o mesmo tamanho que a lista de summaries.")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(args.output_md, labels, summaries)
    write_csv(args.output_csv, labels, summaries)
    print(json.dumps({"output_md": str(args.output_md), "output_csv": str(args.output_csv)}, indent=2))


def write_markdown(path: Path, labels: list[str], summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Comparacao de Runs",
        "",
        "| Metrica | " + " | ".join(labels) + " |",
        "|---|" + "|".join("---:" for _ in labels) + "|",
    ]
    for key, label in METRICS:
        values = [_fmt(summary.get(key)) for summary in summaries]
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Observacoes",
        "",
        "- Compare runs com o mesmo conjunto de tarefas, seed, temperatura e limite de tokens.",
        "- Runs adaptativas reduzem custo, mas o estimador `pass@k` fica menos comparavel em K altos.",
        "- Para steering, esta mesma tabela deve incluir colunas `base`, `best_of_n`, `early_stop` e `steered`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, labels: list[str], summaries: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", *labels])
        for key, label in METRICS:
            writer.writerow([label, *[summary.get(key) for summary in summaries]])


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
