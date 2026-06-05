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

from slm_steering.datasets import DEFAULT_HUMANEVAL_DATASET
from slm_steering.experiment_registry import (
    DECODING_SPECS,
    MODEL_SPECS,
    PRESETS,
    decoding_specs,
    model_specs,
    parse_key_list,
    preset_keys,
    run_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roda uma matriz de modelos e estrategias de decodificacao no HumanEval."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="core",
        help="Atalho experimental. 'core' e recomendado para GPU de 6 GB.",
    )
    parser.add_argument(
        "--models",
        help="Lista separada por virgula para sobrescrever o preset.",
    )
    parser.add_argument(
        "--decodings",
        help="Lista separada por virgula para sobrescrever o preset.",
    )
    parser.add_argument("--list", action="store_true", help="Lista presets, modelos e decodings.")
    parser.add_argument("--dry-run", action="store_true", help="Mostra comandos sem executar.")
    parser.add_argument("--skip-existing", action="store_true", help="Nao reroda summaries existentes.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continua se uma run falhar.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limita o numero de combinacoes.")
    parser.add_argument("--dataset-id", default=DEFAULT_HUMANEVAL_DATASET)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Usa apenas modelos ja presentes no cache local do Hugging Face.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/model_matrix"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print_catalog()
        return

    preset_model_keys, preset_decoding_keys = preset_keys(args.preset)
    model_keys = parse_key_list(args.models) or preset_model_keys
    decoding_keys = parse_key_list(args.decodings) or preset_decoding_keys

    models = model_specs(model_keys)
    decodings = decoding_specs(decoding_keys)
    planned = [(model, decoding) for model in models for decoding in decodings]
    if args.max_runs is not None:
        planned = planned[: args.max_runs]

    manifest_path = args.output_dir / "manifest.json"
    run_entries: list[dict[str, Any]] = []
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, (model, decoding) in enumerate(planned, start=1):
        label = run_label(model, decoding)
        run_prefix = args.output_dir / (
            f"{label}__he_offset{args.offset}_limit{args.limit or 'all'}"
            f"__seed{args.seed}__tok{args.max_new_tokens}"
        )
        jsonl_path = run_prefix.with_name(f"{run_prefix.name}.jsonl")
        summary_path = run_prefix.with_name(f"{run_prefix.name}_summary.json")
        command = build_baseline_command(args, model, decoding, jsonl_path, summary_path)
        entry = base_manifest_entry(args, model, decoding, label, jsonl_path, summary_path)
        entry["command"] = command

        print(f"[{index}/{len(planned)}] {label}", flush=True)
        if args.dry_run:
            entry["status"] = "planned"
            run_entries.append(entry)
            print(" ".join(command), flush=True)
            continue

        if args.skip_existing and summary_path.exists() and jsonl_path.exists():
            entry.update(load_completed_metrics(summary_path))
            entry["status"] = "skipped_existing"
            run_entries.append(entry)
            print(f"  pulando: {summary_path}", flush=True)
            continue

        completed = subprocess.run(command, cwd=ROOT)
        entry["returncode"] = completed.returncode
        if completed.returncode != 0:
            entry["status"] = "failed"
            run_entries.append(entry)
            write_matrix_outputs(args.output_dir, manifest_path, args, run_entries)
            if not args.continue_on_error:
                raise SystemExit(completed.returncode)
            continue

        entry.update(load_completed_metrics(summary_path))
        entry["status"] = "completed"
        annotate_summary(summary_path, entry)
        run_entries.append(entry)
        write_matrix_outputs(args.output_dir, manifest_path, args, run_entries)

    if args.dry_run:
        print("Dry run concluido; nenhum arquivo foi gravado.", flush=True)
        return

    write_matrix_outputs(args.output_dir, manifest_path, args, run_entries)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Resumo CSV: {args.output_dir / 'matrix_summary.csv'}", flush=True)
    print(f"Relatorio: {args.output_dir / 'matrix_report.md'}", flush=True)


def build_baseline_command(
    args: argparse.Namespace,
    model,
    decoding,
    jsonl_path: Path,
    summary_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_baseline.py"),
        "--model-id",
        model.model_id,
        "--dataset-id",
        args.dataset_id,
        "--limit",
        str(args.limit),
        "--offset",
        str(args.offset),
        "--seed",
        str(args.seed),
        "--n",
        str(decoding.n),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(decoding.temperature),
        "--top-p",
        str(decoding.top_p),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--output-jsonl",
        str(jsonl_path),
        "--summary-json",
        str(summary_path),
    ]
    if args.require_cuda:
        command.append("--require-cuda")
    if args.local_files_only:
        command.append("--local-files-only")
    if decoding.early_stop:
        command.append("--early-stop")
    if not model.use_chat_template:
        command.append("--no-chat-template")
    return command


def base_manifest_entry(
    args: argparse.Namespace,
    model,
    decoding,
    label: str,
    jsonl_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    return {
        "label": label,
        "status": "pending",
        "model_key": model.key,
        "model_id": model.model_id,
        "family": model.family,
        "parameters_b": model.parameters_b,
        "gpu_tier": model.gpu_tier,
        "model_notes": model.notes,
        "decoding_key": decoding.key,
        "n": decoding.n,
        "temperature": decoding.temperature,
        "top_p": decoding.top_p,
        "early_stop": decoding.early_stop,
        "decoding_notes": decoding.notes,
        "dataset_id": args.dataset_id,
        "limit": args.limit,
        "offset": args.offset,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
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
    return {"metrics": {key: summary.get(key) for key in keys}}


def annotate_summary(summary_path: Path, entry: dict[str, Any]) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["matrix"] = {
        key: entry[key]
        for key in [
            "label",
            "model_key",
            "model_id",
            "family",
            "parameters_b",
            "gpu_tier",
            "decoding_key",
            "n",
            "temperature",
            "top_p",
            "early_stop",
        ]
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_matrix_outputs(
    output_dir: Path,
    manifest_path: Path,
    args: argparse.Namespace,
    run_entries: list[dict[str, Any]],
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "dataset_id": args.dataset_id,
        "limit": args.limit,
        "offset": args.offset,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "runs": run_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_matrix_csv(output_dir / "matrix_summary.csv", run_entries)
    write_matrix_report(output_dir / "matrix_report.md", run_entries)


def write_matrix_csv(path: Path, run_entries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "status",
        "model_key",
        "family",
        "parameters_b",
        "decoding_key",
        "n",
        "temperature",
        "top_p",
        "early_stop",
        "num_tasks",
        "total_attempts",
        "solved_tasks",
        "strict_pass_at_1",
        "observed_best_of_n",
        "total_generated_tokens",
        "total_generation_seconds",
        "mean_generation_tokens_per_second",
        "token_savings_if_oracle_early_stop",
        "generated_tokens_per_solved_task",
        "effective_tokens_per_solved_task",
        "summary_path",
        "jsonl_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in run_entries:
            metrics = entry.get("metrics", {})
            row = {key: entry.get(key) for key in fieldnames}
            row.update({key: metrics.get(key) for key in metrics if key in fieldnames})
            writer.writerow(row)


def write_matrix_report(path: Path, run_entries: list[dict[str, Any]]) -> None:
    lines = [
        "# Relatorio - Matriz de Modelos",
        "",
        "| Run | Status | Modelo | Decoding | pass@1 | Best-of-N | Tokens | Tokens/s | Economia early stop |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for entry in run_entries:
        metrics = entry.get("metrics", {})
        lines.append(
            "| "
            f"{entry['label']} | "
            f"{entry['status']} | "
            f"{entry['model_key']} | "
            f"{entry['decoding_key']} | "
            f"{_fmt(metrics.get('strict_pass_at_1'))} | "
            f"{_fmt(metrics.get('observed_best_of_n'))} | "
            f"{_fmt(metrics.get('total_generated_tokens'))} | "
            f"{_fmt(metrics.get('mean_generation_tokens_per_second'))} | "
            f"{_fmt(metrics.get('token_savings_if_oracle_early_stop'))} |"
        )
    lines += [
        "",
        "## Leitura Experimental",
        "",
        "- Compare `greedy_n1` com `sample_n5` para separar capacidade imediata de ganho por amostragem.",
        "- Compare `sample_n5` com `sample_n5_early_stop` para medir a utilidade pratica do verificador.",
        "- Compare modelos de tamanho parecido para avaliar se a arquitetura/familia importa alem do numero de parametros.",
        "- Use `conservative_n5` para testar se menor diversidade reduz custo sem destruir acuracia.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def print_catalog() -> None:
    print("Presets:")
    for key, preset in PRESETS.items():
        print(f"  {key}: models={','.join(preset['models'])}; decodings={','.join(preset['decodings'])}")
    print("\nModels:")
    for spec in MODEL_SPECS.values():
        print(f"  {spec.key}: {spec.model_id} ({spec.parameters_b}B, {spec.gpu_tier})")
    print("\nDecodings:")
    for spec in DECODING_SPECS.values():
        print(
            f"  {spec.key}: n={spec.n}, temperature={spec.temperature}, "
            f"top_p={spec.top_p}, early_stop={spec.early_stop}"
        )


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
