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

from slm_steering.benchmark_registry import load_benchmark
from slm_steering.env import assert_cuda_available
from slm_steering.experiment_metrics import extended_generation_metrics
from slm_steering.generation import AutoCodeGenerator, GeneratorConfig
from slm_steering.metrics import summarize
from slm_steering.policies import (
    DifficultyAdaptivePolicy,
    FixedNPolicy,
    LatentScoreAdaptivePolicy,
    LatentScoreConfig,
    PolicyState,
    SteeringAdaptivePolicy,
    VerifierEarlyStopPolicy,
    build_latent_direction_scorer,
)
from slm_steering.verifier import check_correctness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fase 5: Latent-Adaptive Best-of-N.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--benchmark", default="humaneval")
    parser.add_argument("--directions-dir", type=Path, default=Path("runs/phase3/qwen_phase3_probe"))
    parser.add_argument("--steering-config", type=Path, default=None)
    parser.add_argument("--policy", default="latent_adaptive")
    parser.add_argument("--latent-layer", type=int, default=None)
    parser.add_argument("--latent-token-position", default="completion_last")
    parser.add_argument("--max-n", type=int, default=10)
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
    parser.add_argument("--output-dir", type=Path, default=Path("runs/phase5/adaptive_bon"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps({"policy": args.policy, "max_n": args.max_n, "limit": args.limit}, indent=2))
        return
    if args.require_cuda:
        assert_cuda_available()
        if args.device == "auto":
            args.device = "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = build_policy(args.policy, args.max_n)
    generator = AutoCodeGenerator(
        GeneratorConfig(
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            local_files_only=args.local_files_only,
        )
    )
    scorer = None
    if "latent" in args.policy:
        scorer = build_latent_direction_scorer(
            LatentScoreConfig(
                directions_dir=args.directions_dir,
                layer=args.latent_layer,
                token_position=args.latent_token_position,
            )
        )
        if scorer is None:
            print(
                "Aviso: latent_directions.npz nao encontrado ou camada invalida; "
                "a politica latent_adaptive usara apenas priors conservadores.",
                flush=True,
            )
        else:
            print(f"Latent scorer: layer={scorer.layer}, token_position={scorer.token_position}", flush=True)
    problems = load_benchmark(args.benchmark, limit=args.limit, offset=args.offset)
    records = []
    decisions = []
    started_at = datetime.now(timezone.utc).isoformat()
    jsonl_path = args.output_dir / f"{args.policy}_maxn{args.max_n}.jsonl"
    summary_path = args.output_dir / f"{args.policy}_maxn{args.max_n}_summary.json"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for problem_index, problem in enumerate(tqdm(problems, desc=args.policy)):
            record, task_decisions = run_problem(problem, generator, policy, scorer, args, problem_index)
            records.append(record)
            decisions.extend(task_decisions)
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
    summary = summarize(records, requested_n=args.max_n)
    summary.update(extended_generation_metrics(records, summary))
    summary.update(adaptive_metrics(records, fixed_n=args.max_n))
    summary["policy"] = {
        "name": args.policy,
        "max_n": args.max_n,
        "model_id": args.model_id,
        "benchmark": args.benchmark,
        "directions_dir": str(args.directions_dir),
        "latent_layer": scorer.layer if scorer is not None else None,
        "latent_token_position": args.latent_token_position,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / f"{args.policy}_maxn{args.max_n}_decisions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_problem(problem, generator, policy, scorer, args: argparse.Namespace, problem_index: int):
    problem_start = time.perf_counter()
    attempts = []
    decisions = []
    first_success_attempt = None
    for attempt_index in range(1, args.max_n + 1):
        attempt_seed = args.seed + problem_index * args.max_n + attempt_index - 1
        generated = generator.generate(problem.prompt, seed=attempt_seed)
        verified = check_correctness(problem, generated.text, timeout_seconds=args.timeout_seconds)
        latent_score = scorer.score(generator, problem.prompt, generated.text) if scorer is not None else None
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
                "latent_score": latent_score,
                "raw_completion": generated.text,
                "candidate_source": verified.candidate_source,
                "stdout": verified.stdout,
                "stderr": verified.stderr,
            }
        )
        state = PolicyState(
            attempt_index=attempt_index,
            max_attempts=args.max_n,
            solved=first_success_attempt is not None,
            last_passed=verified.passed,
            difficulty=None,
            latent_score=latent_score,
            generated_tokens=sum(item["generated_tokens"] for item in attempts),
            mean_token_cost=max(1.0, sum(item["generated_tokens"] for item in attempts) / len(attempts)),
        )
        decision = policy.decide(state)
        decisions.append(
            {
                "task_id": problem.task_id,
                "attempt": attempt_index,
                "decision": decision.reason.value,
                "continue_sampling": decision.continue_sampling,
                "expected_gain": decision.expected_gain,
            }
        )
        if not decision.continue_sampling:
            break
    record = {
        "task_id": problem.task_id,
        "entry_point": problem.entry_point,
        "prompt": problem.prompt,
        "solved": first_success_attempt is not None,
        "first_success_attempt": first_success_attempt,
        "problem_wall_seconds": time.perf_counter() - problem_start,
        "attempts": attempts,
    }
    return record, decisions


def build_policy(name: str, max_n: int):
    if name == "fixed_n_1":
        return FixedNPolicy(1)
    if name == "fixed_n_5":
        return FixedNPolicy(min(5, max_n))
    if name == "fixed_n_10":
        return FixedNPolicy(min(10, max_n))
    if name == "verifier_early_stop":
        return VerifierEarlyStopPolicy()
    if name == "difficulty_adaptive":
        return DifficultyAdaptivePolicy(default_n=max_n)
    if name == "latent_adaptive":
        return LatentScoreAdaptivePolicy()
    if name in {"steering_fixed_n", "steering_latent_adaptive"}:
        return SteeringAdaptivePolicy()
    raise ValueError(f"policy desconhecida: {name}")


def adaptive_metrics(records: list[dict[str, Any]], *, fixed_n: int) -> dict[str, Any]:
    tokens = [sum(attempt["generated_tokens"] for attempt in record["attempts"]) for record in records]
    attempts = [len(record["attempts"]) for record in records]
    successes = sum(1 for record in records if record["solved"])
    mean_tokens = sum(tokens) / len(tokens) if tokens else 0
    return {
        "success_rate": successes / len(records) if records else 0,
        "mean_tokens": mean_tokens,
        "median_tokens": sorted(tokens)[len(tokens) // 2] if tokens else 0,
        "tokens_per_success": sum(tokens) / successes if successes else None,
        "mean_attempts": sum(attempts) / len(attempts) if attempts else 0,
        "budget_saved_vs_fixed_n": 1 - (sum(attempts) / (len(records) * fixed_n)) if records and fixed_n else 0,
        "accuracy_delta_vs_fixed_n": None,
        "pareto_dominance": None,
    }


if __name__ == "__main__":
    main()
