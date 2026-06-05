from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_run_artifacts(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    prefix: Path,
) -> dict[str, str]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "report_md": prefix.with_name(f"{prefix.name}_report.md"),
        "tasks_csv": prefix.with_name(f"{prefix.name}_tasks.csv"),
        "attempts_csv": prefix.with_name(f"{prefix.name}_attempts.csv"),
        "curve_csv": prefix.with_name(f"{prefix.name}_bon_curve.csv"),
    }
    write_markdown_report(paths["report_md"], records, summary)
    write_tasks_csv(paths["tasks_csv"], records)
    write_attempts_csv(paths["attempts_csv"], records)
    write_bon_curve_csv(paths["curve_csv"], summary)
    return {key: str(path) for key, path in paths.items()}


def write_markdown_report(
    path: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    config = summary.get("config", {})
    lines = [
        "# Relatorio Experimental - Baseline HumanEval",
        "",
        "## Configuracao",
        "",
        "| Campo | Valor |",
        "|---|---|",
    ]
    for key in [
        "model_id",
        "dataset",
        "limit",
        "offset",
        "n",
        "seed",
        "max_new_tokens",
        "temperature",
        "top_p",
        "device",
        "dtype",
        "early_stop",
        "started_at",
        "completed_at",
    ]:
        lines.append(f"| `{key}` | {_fmt(config.get(key))} |")

    metric_rows = [
        ("Tarefas avaliadas", summary.get("num_tasks")),
        ("Tentativas totais", summary.get("total_attempts")),
        ("Tarefas resolvidas", summary.get("solved_tasks")),
        ("pass@1 estrito", summary.get("strict_pass_at_1")),
        ("Best-of-N observado", summary.get("observed_best_of_n")),
        ("Tokens medios por tentativa", summary.get("mean_generated_tokens_per_attempt")),
        ("Tempo medio de geracao por tentativa", summary.get("mean_generation_seconds_per_attempt")),
        ("Tokens/s medio de geracao", summary.get("mean_generation_tokens_per_second")),
        ("Amostras medias ate sucesso/orcamento", summary.get("mean_attempts_until_success_or_budget")),
        ("Tokens efetivos ate sucesso/orcamento", summary.get("mean_tokens_until_success_or_budget")),
        ("Economia potencial de tokens com parada ideal", summary.get("token_savings_if_oracle_early_stop")),
        ("Tokens gerados por tarefa resolvida", summary.get("generated_tokens_per_solved_task")),
    ]
    lines += [
        "",
        "## Metricas Principais",
        "",
        "| Metrica | Valor |",
        "|---|---:|",
    ]
    for label, value in metric_rows:
        lines.append(f"| {label} | {_fmt(value)} |")

    lines += [
        "",
        "## Curva Best-of-K",
        "",
        "| K | Best-of-K observado | pass@K estimado | Tarefas elegiveis |",
        "|---:|---:|---:|---:|",
    ]
    observed = summary.get("observed_best_of_k", {})
    estimated = summary.get("pass_at_k_estimate", {})
    eligible = summary.get("pass_at_k_eligible_tasks", {})
    for key in sorted(observed, key=lambda item: int(item)):
        lines.append(
            f"| {key} | {_fmt(observed.get(key))} | {_fmt(estimated.get(key))} | {_fmt(eligible.get(key))} |"
        )

    lines += [
        "",
        "## Tarefas",
        "",
        "| Task | Resolvida | Primeira correta | Tentativas | Tokens | Segundos |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        attempts = record["attempts"]
        tokens = sum(attempt["generated_tokens"] for attempt in attempts)
        seconds = sum(
            attempt["generation_seconds"] + attempt["verification_seconds"]
            for attempt in attempts
        )
        lines.append(
            "| "
            f"{record['task_id']} | "
            f"{record['solved']} | "
            f"{_fmt(record['first_success_attempt'])} | "
            f"{len(attempts)} | "
            f"{tokens} | "
            f"{_fmt(seconds)} |"
        )

    lines += [
        "",
        "## Leitura Para o TCC",
        "",
        "- `strict_pass_at_1` mede a capacidade imediata do SLM sem escalonamento de inferencia.",
        "- `observed_best_of_k` mostra quanto ganho vem apenas de amostrar mais respostas.",
        "- `mean_attempts_until_success_or_budget` aproxima o custo medio de uma politica de parada com verificador.",
        "- `token_savings_if_oracle_early_stop` estima a economia maxima que um verificador perfeito poderia capturar no mesmo conjunto.",
        "- Esta etapa define a linha de base que o futuro activation steering precisa superar em acuracia, latencia e tokens.",
    ]
    if summary.get("adaptive_sampling_used"):
        lines.append(
            "- Esta run usou amostragem adaptativa; interprete `pass_at_k_estimate` apenas para K com tarefas elegiveis suficientes."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tasks_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_id",
                "entry_point",
                "solved",
                "first_success_attempt",
                "attempts",
                "generated_tokens",
                "generation_seconds",
                "verification_seconds",
                "problem_wall_seconds",
            ],
        )
        writer.writeheader()
        for record in records:
            attempts = record["attempts"]
            writer.writerow(
                {
                    "task_id": record["task_id"],
                    "entry_point": record["entry_point"],
                    "solved": record["solved"],
                    "first_success_attempt": record["first_success_attempt"],
                    "attempts": len(attempts),
                    "generated_tokens": sum(a["generated_tokens"] for a in attempts),
                    "generation_seconds": sum(a["generation_seconds"] for a in attempts),
                    "verification_seconds": sum(a["verification_seconds"] for a in attempts),
                    "problem_wall_seconds": record["problem_wall_seconds"],
                }
            )


def write_attempts_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_id",
                "entry_point",
                "attempt",
                "seed",
                "passed",
                "generated_tokens",
                "generation_seconds",
                "verification_seconds",
                "tokens_per_second",
                "error_type",
            ],
        )
        writer.writeheader()
        for record in records:
            for attempt in record["attempts"]:
                generation_seconds = attempt["generation_seconds"]
                writer.writerow(
                    {
                        "task_id": record["task_id"],
                        "entry_point": record["entry_point"],
                        "attempt": attempt["attempt"],
                        "seed": attempt["seed"],
                        "passed": attempt["passed"],
                        "generated_tokens": attempt["generated_tokens"],
                        "generation_seconds": generation_seconds,
                        "verification_seconds": attempt["verification_seconds"],
                        "tokens_per_second": (
                            attempt["generated_tokens"] / generation_seconds
                            if generation_seconds > 0
                            else None
                        ),
                        "error_type": attempt["error_type"],
                    }
                )


def write_bon_curve_csv(path: Path, summary: dict[str, Any]) -> None:
    observed = summary.get("observed_best_of_k", {})
    estimated = summary.get("pass_at_k_estimate", {})
    eligible = summary.get("pass_at_k_eligible_tasks", {})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "k",
                "observed_best_of_k",
                "pass_at_k_estimate",
                "eligible_tasks",
            ],
        )
        writer.writeheader()
        for key in sorted(observed, key=lambda item: int(item)):
            writer.writerow(
                {
                    "k": int(key),
                    "observed_best_of_k": observed.get(key),
                    "pass_at_k_estimate": estimated.get(key),
                    "eligible_tasks": eligible.get(key),
                }
            )


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
