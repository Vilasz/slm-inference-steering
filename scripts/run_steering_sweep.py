from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fase 4: causal activation steering sweep.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--benchmark", default="humaneval")
    parser.add_argument("--directions-dir", type=Path, default=Path("runs/phase3/qwen_phase3_probe"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/phase4/steering_sweep"))
    parser.add_argument("--layers", default="12")
    parser.add_argument("--alphas", default="0,1")
    parser.add_argument(
        "--direction-types",
        default="correctness_direction,negative_correctness_direction,random_direction",
    )
    parser.add_argument("--token-selection", default="last")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_torch_runtime(args)

    from slm_steering.activations.layer_sweep import (
        build_sweep,
        parse_float_list,
        parse_int_list,
        parse_str_list,
    )

    layers = parse_int_list(args.layers)
    alphas = parse_float_list(args.alphas)
    direction_types = parse_str_list(args.direction_types)
    sweep = build_sweep(layers, alphas, direction_types)
    if args.dry_run:
        for point in sweep:
            print(point.label)
        return

    from slm_steering.benchmark_registry import load_benchmark

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "steering_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_entries = []
    generator = None
    problems = None
    for point in sweep:
        run_prefix = args.output_dir / point.label
        jsonl_path = run_prefix.with_suffix(".jsonl")
        summary_path = run_prefix.with_name(f"{run_prefix.name}_summary.json")
        print(point.label, flush=True)
        if args.skip_existing and jsonl_path.exists() and summary_path.exists():
            manifest_entries.append({"label": point.label, "status": "skipped_existing", "summary_path": str(summary_path), "jsonl_path": str(jsonl_path)})
            continue
        if generator is None:
            generator = build_generator(args, point.layer)
            problems = load_benchmark(args.benchmark, limit=args.limit, offset=args.offset)
        entry = run_single_sweep_point(args, point, generator, problems or [], jsonl_path, summary_path)
        manifest_entries.append(entry)
        write_manifest(args.output_dir, args, manifest_entries)
    write_manifest(args.output_dir, args, manifest_entries)


def prepare_torch_runtime(args: argparse.Namespace) -> None:
    """Import/check torch before Windows-hostile heavy module imports."""
    if args.dry_run:
        return
    if args.require_cuda or args.device == "cuda":
        from slm_steering.env import assert_cuda_available

        assert_cuda_available()
        if args.device == "auto":
            args.device = "cuda"
        return

    # A real generation run will need torch even in auto/CPU mode. Importing it
    # here keeps Windows DLL initialization ahead of pandas/datasets/transformers.
    import torch  # noqa: F401


def build_generator(args: argparse.Namespace, initial_layer: int) -> SteeredCodeGenerator:
    from slm_steering.activations.steering import SteeredCodeGenerator, SteeringConfig
    from slm_steering.generation import GeneratorConfig

    return SteeredCodeGenerator(
        GeneratorConfig(
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            local_files_only=args.local_files_only,
        ),
        SteeringConfig(layers=[initial_layer], direction_by_layer={initial_layer: [0.0]}, alpha=0.0),
    )


def run_single_sweep_point(
    args: argparse.Namespace,
    point,
    generator: SteeredCodeGenerator,
    problems: list[Any],
    jsonl_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    from slm_steering.activations.controls import load_direction_map
    from slm_steering.activations.steering import SteeringConfig
    from slm_steering.experiment_metrics import extended_generation_metrics
    from slm_steering.metrics import summarize

    direction_map = load_direction_map(
        args.directions_dir,
        direction_type=point.direction_type,
        layers=[point.layer],
        hidden_size=generator.hidden_size,
        seed=args.seed,
        allow_missing=True,
    )
    generator.steering_config = SteeringConfig(
        layers=[point.layer],
        direction_by_layer=direction_map,
        alpha=point.alpha,
        token_selection=args.token_selection,
        direction_type=point.direction_type,
    )
    records = []
    started_at = datetime.now(timezone.utc).isoformat()
    with jsonl_path.open("w", encoding="utf-8") as file:
        for problem_index, problem in enumerate(tqdm(problems, desc=point.label)):
            record = run_problem(problem, generator, args.n, args.seed, problem_index * args.n, args.timeout_seconds)
            records.append(record)
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
    summary = summarize(records, requested_n=args.n)
    summary.update(extended_generation_metrics(records, summary))
    summary["steering"] = {
        "label": point.label,
        "model_id": args.model_id,
        "benchmark": args.benchmark,
        "layer": point.layer,
        "alpha": point.alpha,
        "direction_type": point.direction_type,
        "directions_dir": str(args.directions_dir),
        "token_selection": args.token_selection,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "label": point.label,
        "status": "completed",
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "metrics": {
            "pass_at_1": summary.get("strict_pass_at_1"),
            "solved_rate": summary.get("observed_best_of_n"),
            "mean_attempts_to_success": summary.get("mean_attempts_until_success_or_budget"),
            "tokens_until_first_success": summary.get("mean_tokens_until_success_or_budget"),
            "diversity_score": summary.get("diversity_score"),
            "collapse_score": summary.get("collapse_score"),
        },
    }


def run_problem(problem, generator, n: int, seed: int, global_attempt_start: int, timeout_seconds: float) -> dict[str, Any]:
    from slm_steering.verifier import check_correctness

    problem_start = time.perf_counter()
    attempts = []
    first_success_attempt = None
    for attempt_index in range(1, n + 1):
        attempt_seed = seed + global_attempt_start + attempt_index - 1
        generated = generator.generate(problem.prompt, seed=attempt_seed)
        verified = check_correctness(problem, generated.text, timeout_seconds=timeout_seconds)
        if verified.passed and first_success_attempt is None:
            first_success_attempt = attempt_index
        attempts.append(
            {
                "attempt": attempt_index,
                "seed": attempt_seed,
                "passed": verified.passed,
                "generated_tokens": generated.generated_tokens,
                "generation_seconds": generated.generation_seconds,
                "verification_seconds": verified.verification_seconds,
                "error_type": verified.error_type,
                "raw_completion": generated.text,
                "candidate_source": verified.candidate_source,
                "stdout": verified.stdout,
                "stderr": verified.stderr,
            }
        )
    return {
        "task_id": problem.task_id,
        "entry_point": problem.entry_point,
        "prompt": problem.prompt,
        "solved": first_success_attempt is not None,
        "first_success_attempt": first_success_attempt,
        "problem_wall_seconds": time.perf_counter() - problem_start,
        "attempts": attempts,
    }


def write_manifest(output_dir: Path, args: argparse.Namespace, entries: list[dict[str, Any]]) -> None:
    manifest = {
        "phase": "phase4_causal_activation_steering",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "benchmark": args.benchmark,
        "directions_dir": str(args.directions_dir),
        "runs": entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
