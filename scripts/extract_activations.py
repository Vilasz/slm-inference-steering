from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slm_steering.env import assert_cuda_available, collect_torch_environment, cuda_install_hint, format_torch_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fase 3: extrai ativacoes internas de tentativas ja geradas."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--layers", default="0:28:4", help="Ex.: 0,8,16 ou 0:28:4 ou all.")
    parser.add_argument(
        "--token-position",
        default="completion_last",
        help="last, completion_last, prompt_last, first_completion ou index:<n>.",
    )
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument(
        "--difficulty-filter",
        default="sampling_sensitive,fragile",
        help="Classes separadas por virgula; use 'all' para nao filtrar.",
    )
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-attempts-per-task", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--activation-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary_json)
    model_id = args.model_id or _summary_value(summary, "model_id")
    if not model_id:
        raise ValueError("--model-id e obrigatorio quando o summary nao contem config/phase2.model_id")
    benchmark = args.benchmark or _summary_value(summary, "benchmark") or "humaneval"
    if args.require_cuda or args.device == "cuda":
        assert_cuda_available()
        if args.device == "auto":
            args.device = "cuda"
    elif not args.dry_run:
        torch_env = collect_torch_environment()
        if not torch_env.torch_installed:
            raise RuntimeError(
                "Nao foi possivel importar torch antes da extracao de ativacoes.\n\n"
                f"{format_torch_environment(torch_env)}\n\n"
                f"{cuda_install_hint()}"
            )

    # On Windows, import torch before pandas/datasets/transformers-heavy modules.
    # This avoids intermittent DLL initialization failures around torch/lib/c10.dll.
    from slm_steering.activations.extraction import (
        ActivationExtractionConfig,
        ActivationExtractor,
        plan_activation_extraction,
    )

    config = ActivationExtractionConfig(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        model_id=model_id,
        layer_spec=args.layers,
        token_position=args.token_position,
        benchmark=benchmark,
        dataset_id=args.dataset_id,
        device=args.device,
        dtype=args.dtype,
        use_chat_template=not args.no_chat_template,
        local_files_only=args.local_files_only,
        difficulty_filter=tuple(parse_csv(args.difficulty_filter)),
        task_ids=tuple(parse_csv(args.task_ids)),
        max_tasks=args.max_tasks,
        max_attempts_per_task=args.max_attempts_per_task,
        activation_dtype=args.activation_dtype,
    )
    if args.dry_run:
        print(json.dumps(plan_activation_extraction(config), indent=2, ensure_ascii=False))
        return
    extractor = ActivationExtractor(config)
    paths = extractor.extract()
    print(json.dumps(paths, indent=2, ensure_ascii=False))


def load_summary(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_value(summary: dict, key: str):
    for section in ["phase2", "matrix", "config"]:
        value = summary.get(section, {}).get(key)
        if value is not None:
            return value
    return None


def parse_csv(value: str) -> list[str]:
    if value.strip().lower() in {"", "all", "none", "*"}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
