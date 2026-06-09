from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slm_steering.benchmark_registry import list_benchmarks
from slm_steering.experiment_registry import MODEL_SPECS, safe_name


DEFAULT_MODELS = "qwen2.5-coder-0.5b-instruct,qwen2.5-coder-1.5b-instruct"
DEFAULT_TEMPERATURES = "0.2,0.6,0.8,1.0"
DEFAULT_N_VALUES = "1,2,5,10"
MODEL_SLUGS = {
    "qwen2.5-coder-0.5b-instruct": "qwen05b",
    "qwen2.5-coder-1.5b-instruct": "qwen15b",
    "deepseek-coder-1.3b-instruct": "deepseek13b",
    "qwen2.5-coder-3b-instruct": "qwen3b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fase 2: matriz difficulty-aware de modelos, temperatura e Best-of-N."
    )
    parser.add_argument("--list", action="store_true", help="Lista modelos conhecidos e benchmarks.")
    parser.add_argument("--models", default=DEFAULT_MODELS, help="Chaves ou IDs HF separados por virgula.")
    parser.add_argument("--temperatures", default=DEFAULT_TEMPERATURES, help="Temperaturas separadas por virgula.")
    parser.add_argument("--n-values", default=DEFAULT_N_VALUES, help="Valores de N separados por virgula.")
    parser.add_argument(
        "--early-stop-modes",
        default="full",
        help="Modos separados por virgula: full,early_stop.",
    )
    parser.add_argument("--benchmark", default="humaneval")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/phase2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print_catalog()
        return

    models = [resolve_model(value) for value in parse_csv(args.models)]
    temperatures = parse_float_csv(args.temperatures)
    n_values = parse_int_csv(args.n_values)
    early_stop_modes = parse_csv(args.early_stop_modes)
    planned = list(iter_experiment_configs(models, temperatures, n_values, early_stop_modes))
    if args.max_runs is not None:
        planned = planned[: args.max_runs]

    manifest_path = args.output_dir / "manifest.json"
    entries: list[dict[str, Any]] = []
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, config in enumerate(planned, start=1):
        label = run_label(config)
        jsonl_path = args.output_dir / f"{label}.jsonl"
        summary_path = args.output_dir / f"{label}_summary.json"
        command = build_baseline_command(args, config, jsonl_path, summary_path)
        entry = manifest_entry(args, config, label, jsonl_path, summary_path, command)

        print(f"[{index}/{len(planned)}] {label}", flush=True)
        if args.dry_run:
            entry["status"] = "planned"
            entries.append(entry)
            print(" ".join(command), flush=True)
            continue

        if args.skip_existing and jsonl_path.exists() and summary_path.exists():
            entry["status"] = "skipped_existing"
            entry["metrics"] = load_completed_metrics(summary_path)
            entries.append(entry)
            print(f"  pulando: {summary_path}", flush=True)
            continue

        completed = subprocess.run(command, cwd=ROOT)
        entry["returncode"] = completed.returncode
        if completed.returncode != 0:
            entry["status"] = "failed"
            entries.append(entry)
            write_phase2_outputs(args.output_dir, manifest_path, args, entries)
            if not args.continue_on_error:
                raise SystemExit(completed.returncode)
            continue

        entry["status"] = "completed"
        entry["metrics"] = load_completed_metrics(summary_path)
        annotate_summary(summary_path, entry)
        entries.append(entry)
        write_phase2_outputs(args.output_dir, manifest_path, args, entries)

    if args.dry_run:
        print("Dry run concluido; nenhum arquivo foi gravado.", flush=True)
        return

    write_phase2_outputs(args.output_dir, manifest_path, args, entries)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Resumo CSV: {args.output_dir / 'phase2_summary.csv'}", flush=True)
    print(f"Relatorio: {args.output_dir / 'phase2_report.md'}", flush=True)


def iter_experiment_configs(
    models: list[dict[str, Any]],
    temperatures: list[float],
    n_values: list[int],
    early_stop_modes: list[str],
):
    for model in models:
        for temperature in temperatures:
            for n in n_values:
                for mode in early_stop_modes:
                    if mode not in {"full", "early_stop"}:
                        raise ValueError("--early-stop-modes aceita apenas full,early_stop")
                    if mode == "early_stop" and n == 1:
                        continue
                    yield {
                        "model_key": model["key"],
                        "model_id": model["model_id"],
                        "model_slug": model["slug"],
                        "use_chat_template": model["use_chat_template"],
                        "temperature": temperature,
                        "n": n,
                        "early_stop": mode == "early_stop",
                    }


def build_baseline_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    jsonl_path: Path,
    summary_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_baseline.py"),
        "--benchmark",
        args.benchmark,
        "--model-id",
        config["model_id"],
        "--limit",
        str(args.limit),
        "--offset",
        str(args.offset),
        "--seed",
        str(args.seed),
        "--n",
        str(config["n"]),
        "--temperature",
        str(config["temperature"]),
        "--top-p",
        str(args.top_p),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--output-jsonl",
        str(jsonl_path),
        "--summary-json",
        str(summary_path),
    ]
    if args.dataset_id:
        command.extend(["--dataset-id", args.dataset_id])
    if args.require_cuda:
        command.append("--require-cuda")
    if args.local_files_only:
        command.append("--local-files-only")
    if config["early_stop"]:
        command.append("--early-stop")
    if not config["use_chat_template"]:
        command.append("--no-chat-template")
    return command


def manifest_entry(
    args: argparse.Namespace,
    config: dict[str, Any],
    label: str,
    jsonl_path: Path,
    summary_path: Path,
    command: list[str],
) -> dict[str, Any]:
    return {
        "label": label,
        "status": "pending",
        "benchmark": args.benchmark,
        "dataset_id": args.dataset_id,
        "model_key": config["model_key"],
        "model_id": config["model_id"],
        "temperature": config["temperature"],
        "top_p": args.top_p,
        "n": config["n"],
        "early_stop": config["early_stop"],
        "limit": args.limit,
        "offset": args.offset,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "command": command,
    }


def load_completed_metrics(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    keys = [
        "num_tasks",
        "total_attempts",
        "solved_tasks",
        "strict_pass_at_1",
        "observed_best_of_n",
        "mean_generated_tokens_per_attempt",
        "total_generated_tokens",
        "mean_generation_seconds_per_attempt",
        "total_generation_seconds",
        "mean_generation_tokens_per_second",
        "mean_attempts_until_success_or_budget",
        "mean_tokens_until_success_or_budget",
        "total_tokens_until_success_or_budget",
        "token_savings_if_oracle_early_stop",
        "generated_tokens_per_solved_task",
        "effective_tokens_per_solved_task",
    ]
    return {key: summary.get(key) for key in keys}


def annotate_summary(summary_path: Path, entry: dict[str, Any]) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["phase2"] = {
        key: entry[key]
        for key in [
            "label",
            "benchmark",
            "dataset_id",
            "model_key",
            "model_id",
            "temperature",
            "top_p",
            "n",
            "early_stop",
            "limit",
            "offset",
            "seed",
            "max_new_tokens",
        ]
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_phase2_outputs(
    output_dir: Path,
    manifest_path: Path,
    args: argparse.Namespace,
    entries: list[dict[str, Any]],
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": args.benchmark,
        "limit": args.limit,
        "offset": args.offset,
        "seed": args.seed,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "runs": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_phase2_csv(output_dir / "phase2_summary.csv", entries)
    write_phase2_report(output_dir / "phase2_report.md", entries)


def write_phase2_csv(path: Path, entries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "status",
        "benchmark",
        "model_key",
        "model_id",
        "temperature",
        "top_p",
        "n",
        "early_stop",
        "num_tasks",
        "total_attempts",
        "solved_tasks",
        "strict_pass_at_1",
        "observed_best_of_n",
        "total_generated_tokens",
        "mean_tokens_until_success_or_budget",
        "token_savings_if_oracle_early_stop",
        "summary_path",
        "jsonl_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = {key: entry.get(key) for key in fieldnames}
            metrics = entry.get("metrics", {})
            row.update({key: metrics.get(key) for key in metrics if key in fieldnames})
            writer.writerow(row)


def write_phase2_report(path: Path, entries: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 2 - Difficulty-Aware Inference Scaling",
        "",
        "| Run | Status | Modelo | Temp | N | Early stop | pass@1 | Best-of-N | Tokens efetivos |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in entries:
        metrics = entry.get("metrics", {})
        lines.append(
            "| "
            f"{entry['label']} | "
            f"{entry['status']} | "
            f"{entry['model_key']} | "
            f"{entry['temperature']} | "
            f"{entry['n']} | "
            f"{entry['early_stop']} | "
            f"{_fmt(metrics.get('strict_pass_at_1'))} | "
            f"{_fmt(metrics.get('observed_best_of_n'))} | "
            f"{_fmt(metrics.get('mean_tokens_until_success_or_budget'))} |"
        )
    lines += [
        "",
        "## Leitura",
        "",
        "- `marginal_gain_k` mostra quanto cada tentativa adicional compra em acuracia.",
        "- A fronteira de Pareto compara custo medio efetivo e taxa de resolucao.",
        "- Classes de dificuldade ajudam a identificar quando Best-of-N realmente agrega valor.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_label(config: dict[str, Any]) -> str:
    label = f"{config['model_slug']}_temp{temperature_slug(config['temperature'])}_n{config['n']}"
    if config["early_stop"]:
        label += "_earlystop"
    return label


def resolve_model(value: str) -> dict[str, Any]:
    if value in MODEL_SPECS:
        spec = MODEL_SPECS[value]
        return {
            "key": spec.key,
            "model_id": spec.model_id,
            "slug": MODEL_SLUGS.get(spec.key, safe_name(spec.key)),
            "use_chat_template": spec.use_chat_template,
        }
    return {
        "key": safe_name(value),
        "model_id": value,
        "slug": safe_name(value).replace(".", ""),
        "use_chat_template": True,
    }


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def parse_int_csv(value: str) -> list[int]:
    parsed = [int(item) for item in parse_csv(value)]
    if any(item < 1 for item in parsed):
        raise ValueError("--n-values deve conter apenas inteiros >= 1")
    return parsed


def temperature_slug(value: float) -> str:
    return f"{int(round(value * 10)):02d}"


def print_catalog() -> None:
    print("Modelos conhecidos:")
    for key, spec in MODEL_SPECS.items():
        print(f"  {key}: {spec.model_id}")
    print("\nBenchmarks:")
    for spec in list_benchmarks():
        print(f"  {spec.key}: {spec.name} [{spec.status}] - {spec.notes}")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
