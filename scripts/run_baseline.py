from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slm_steering.datasets import DEFAULT_HUMANEVAL_DATASET, CodingProblem, load_humaneval
from slm_steering.env import assert_cuda_available, collect_torch_environment, format_torch_environment
from slm_steering.generation import AutoCodeGenerator, GeneratorConfig
from slm_steering.metrics import summarize
from slm_steering.reporting import write_run_artifacts
from slm_steering.verifier import check_correctness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline: prompt -> modelo de codigo -> resposta -> verificador -> metricas."
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Modelo Hugging Face a ser avaliado.",
    )
    parser.add_argument("--n", type=int, default=1, help="Numero de respostas por problema.")
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_HUMANEVAL_DATASET,
        help="Dataset Hugging Face no formato namespace/name.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limite de problemas do HumanEval.")
    parser.add_argument("--offset", type=int, default=0, help="Offset inicial no HumanEval.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed base.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="auto usa CUDA quando disponivel.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Falha antes de carregar o modelo se CUDA nao estiver disponivel.",
    )
    parser.add_argument(
        "--diagnose-env",
        action="store_true",
        help="Mostra informacoes de Python/PyTorch/CUDA e encerra.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Usa o prompt bruto do HumanEval, util para modelos base nao-instruct.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Carrega modelo/tokenizer apenas do cache local do Hugging Face.",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Para de gerar para um problema assim que o verificador aprovar uma resposta.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5.0,
        help="Timeout do verificador por tentativa.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("runs/baseline_humaneval.jsonl"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("runs/baseline_humaneval_summary.json"),
    )
    parser.add_argument(
        "--skip-analysis-artifacts",
        action="store_true",
        help="Nao gera Markdown/CSVs auxiliares ao final.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 1:
        raise ValueError("--n deve ser >= 1")
    if args.diagnose_env:
        print(format_torch_environment(collect_torch_environment()))
        return
    if args.require_cuda or args.device == "cuda":
        assert_cuda_available()
    if args.require_cuda and args.device == "auto":
        args.device = "cuda"

    started_at = datetime.now(timezone.utc).isoformat()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    generator = AutoCodeGenerator(
        GeneratorConfig(
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_chat_template=not args.no_chat_template,
            local_files_only=args.local_files_only,
        )
    )
    problems = load_humaneval(limit=args.limit, offset=args.offset, dataset_id=args.dataset_id)

    records: list[dict] = []
    with args.output_jsonl.open("w", encoding="utf-8") as output_file:
        for problem_index, problem in enumerate(tqdm(problems, desc="HumanEval")):
            record = run_problem(
                problem=problem,
                generator=generator,
                n=args.n,
                seed=args.seed,
                global_attempt_start=problem_index * args.n,
                timeout_seconds=args.timeout_seconds,
                early_stop=args.early_stop,
            )
            records.append(record)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()

    summary = summarize(records, requested_n=args.n)
    summary["config"] = {
        "model_id": args.model_id,
        "dataset": f"{args.dataset_id}/test",
        "limit": args.limit,
        "offset": args.offset,
        "n": args.n,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "device": generator.device,
        "dtype": str(generator.dtype).replace("torch.", ""),
        "chat_template": not args.no_chat_template,
        "local_files_only": args.local_files_only,
        "early_stop": args.early_stop,
        "timeout_seconds": args.timeout_seconds,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if generator.device == "cuda":
        summary["cuda"] = generator.cuda_memory_stats()

    args.summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not args.skip_analysis_artifacts:
        artifact_paths = write_run_artifacts(records, summary, args.output_jsonl.with_suffix(""))
        summary["artifacts"] = artifact_paths
        args.summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_problem(
    problem: CodingProblem,
    generator: AutoCodeGenerator,
    n: int,
    seed: int,
    global_attempt_start: int,
    timeout_seconds: float,
    early_stop: bool,
) -> dict:
    problem_start = time.perf_counter()
    attempts = []
    first_success_attempt = None

    for attempt_index in range(1, n + 1):
        attempt_seed = seed + global_attempt_start + attempt_index - 1
        generated = generator.generate(problem.prompt, seed=attempt_seed)
        verified = check_correctness(problem, generated.text, timeout_seconds=timeout_seconds)

        passed = verified.passed
        if passed and first_success_attempt is None:
            first_success_attempt = attempt_index

        attempts.append(
            {
                "attempt": attempt_index,
                "seed": attempt_seed,
                "passed": passed,
                "generated_tokens": generated.generated_tokens,
                "generation_seconds": generated.generation_seconds,
                "tokens_per_second": (
                    generated.generated_tokens / generated.generation_seconds
                    if generated.generation_seconds > 0
                    else None
                ),
                "verification_seconds": verified.verification_seconds,
                "error_type": verified.error_type,
                "raw_completion": generated.text,
                "candidate_source": verified.candidate_source,
                "stdout": verified.stdout,
                "stderr": verified.stderr,
            }
        )

        if early_stop and passed:
            break

    return {
        "task_id": problem.task_id,
        "entry_point": problem.entry_point,
        "solved": first_success_attempt is not None,
        "first_success_attempt": first_success_attempt,
        "problem_wall_seconds": time.perf_counter() - problem_start,
        "attempts": attempts,
    }


if __name__ == "__main__":
    main()
