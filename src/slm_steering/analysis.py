from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from slm_steering.reporting import read_jsonl


@dataclass(frozen=True)
class RunBundle:
    label: str
    records: list[dict[str, Any]]
    summary: dict[str, Any]


def load_run(jsonl_path: Path, summary_path: Path | None = None, label: str | None = None) -> RunBundle:
    records = read_jsonl(jsonl_path)
    summary = {}
    if summary_path and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return RunBundle(label=label or jsonl_path.stem, records=records, summary=summary)


def attempts_frame(run: RunBundle) -> pd.DataFrame:
    rows = []
    for record in run.records:
        for attempt in record["attempts"]:
            generation_seconds = attempt["generation_seconds"]
            rows.append(
                {
                    "run": run.label,
                    "task_id": record["task_id"],
                    "entry_point": record["entry_point"],
                    "attempt": attempt["attempt"],
                    "seed": attempt["seed"],
                    "passed": bool(attempt["passed"]),
                    "generated_tokens": attempt["generated_tokens"],
                    "generation_seconds": generation_seconds,
                    "verification_seconds": attempt["verification_seconds"],
                    "total_seconds": generation_seconds + attempt["verification_seconds"],
                    "tokens_per_second": (
                        attempt["generated_tokens"] / generation_seconds
                        if generation_seconds > 0
                        else np.nan
                    ),
                    "error_type": attempt.get("error_type") or "passed",
                    "raw_chars": len(attempt.get("raw_completion", "")),
                    "candidate_lines": len(attempt.get("candidate_source", "").splitlines()),
                }
            )
    return pd.DataFrame(rows)


def tasks_frame(run: RunBundle) -> pd.DataFrame:
    rows = []
    for record in run.records:
        attempts = record["attempts"]
        correct_count = sum(1 for attempt in attempts if attempt["passed"])
        generated_tokens = sum(attempt["generated_tokens"] for attempt in attempts)
        total_seconds = sum(
            attempt["generation_seconds"] + attempt["verification_seconds"]
            for attempt in attempts
        )
        rows.append(
            {
                "run": run.label,
                "task_id": record["task_id"],
                "entry_point": record["entry_point"],
                "solved": bool(record["solved"]),
                "first_success_attempt": record["first_success_attempt"],
                "attempts": len(attempts),
                "correct_count": correct_count,
                "solution_rate": correct_count / len(attempts) if attempts else np.nan,
                "generated_tokens": generated_tokens,
                "total_seconds": total_seconds,
                "problem_wall_seconds": record["problem_wall_seconds"],
                "difficulty": _difficulty_label(correct_count, len(attempts)),
            }
        )
    return pd.DataFrame(rows)


def comparison_frame(runs: list[RunBundle]) -> pd.DataFrame:
    rows = []
    for run in runs:
        summary = run.summary
        rows.append(
            {
                "run": run.label,
                "tasks": summary.get("num_tasks"),
                "requested_n": summary.get("requested_n"),
                "adaptive": summary.get("adaptive_sampling_used"),
                "attempts": summary.get("total_attempts"),
                "solved": summary.get("solved_tasks"),
                "pass_at_1": summary.get("strict_pass_at_1"),
                "best_of_n": summary.get("observed_best_of_n"),
                "generated_tokens": summary.get("total_generated_tokens"),
                "tokens_to_success_or_budget": summary.get("total_tokens_until_success_or_budget"),
                "tokens_per_solved_task": summary.get("generated_tokens_per_solved_task"),
                "effective_tokens_per_solved_task": summary.get("effective_tokens_per_solved_task"),
                "generation_seconds": summary.get("total_generation_seconds"),
                "mean_tokens_per_second": summary.get("mean_generation_tokens_per_second"),
                "oracle_early_stop_token_savings": summary.get("token_savings_if_oracle_early_stop"),
            }
        )
    return pd.DataFrame(rows)


def bon_curve_frame(run: RunBundle) -> pd.DataFrame:
    summary = run.summary
    observed = summary.get("observed_best_of_k", {})
    estimated = summary.get("pass_at_k_estimate", {})
    eligible = summary.get("pass_at_k_eligible_tasks", {})
    rows = []
    for key in sorted(observed, key=lambda item: int(item)):
        rows.append(
            {
                "run": run.label,
                "k": int(key),
                "observed_best_of_k": observed.get(key),
                "pass_at_k_estimate": estimated.get(key),
                "eligible_tasks": eligible.get(key),
            }
        )
    return pd.DataFrame(rows)


def counterfactual_early_stop_frame(run: RunBundle) -> pd.DataFrame:
    rows = []
    for record in run.records:
        attempts = record["attempts"]
        cutoff = record["first_success_attempt"] or len(attempts)
        full_tokens = sum(attempt["generated_tokens"] for attempt in attempts)
        effective_tokens = sum(attempt["generated_tokens"] for attempt in attempts[:cutoff])
        full_seconds = sum(
            attempt["generation_seconds"] + attempt["verification_seconds"]
            for attempt in attempts
        )
        effective_seconds = sum(
            attempt["generation_seconds"] + attempt["verification_seconds"]
            for attempt in attempts[:cutoff]
        )
        rows.append(
            {
                "run": run.label,
                "task_id": record["task_id"],
                "solved": bool(record["solved"]),
                "full_attempts": len(attempts),
                "effective_attempts": cutoff,
                "saved_attempts": len(attempts) - cutoff,
                "full_tokens": full_tokens,
                "effective_tokens": effective_tokens,
                "saved_tokens": full_tokens - effective_tokens,
                "token_savings_rate": (
                    1 - effective_tokens / full_tokens if full_tokens else np.nan
                ),
                "full_seconds": full_seconds,
                "effective_seconds": effective_seconds,
                "saved_seconds": full_seconds - effective_seconds,
            }
        )
    return pd.DataFrame(rows)


def diversity_frame(run: RunBundle) -> pd.DataFrame:
    rows = []
    for record in run.records:
        attempts = record["attempts"]
        sources = [attempt.get("candidate_source", "") for attempt in attempts]
        distances = [
            1 - SequenceMatcher(None, left, right).ratio()
            for left, right in combinations(sources, 2)
        ]
        correct_count = sum(1 for attempt in attempts if attempt["passed"])
        rows.append(
            {
                "run": run.label,
                "task_id": record["task_id"],
                "attempts": len(attempts),
                "solved": bool(record["solved"]),
                "correct_count": correct_count,
                "solution_rate": correct_count / len(attempts) if attempts else np.nan,
                "mean_pairwise_code_distance": float(np.mean(distances)) if distances else np.nan,
                "max_pairwise_code_distance": float(np.max(distances)) if distances else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(
    values: list[float] | pd.Series,
    *,
    n_boot: int = 5000,
    confidence: float = 0.95,
    seed: int = 1234,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)]
    if len(array) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_boot, len(array)), replace=True).mean(axis=1)
    alpha = 1 - confidence
    return (
        float(array.mean()),
        float(np.quantile(samples, alpha / 2)),
        float(np.quantile(samples, 1 - alpha / 2)),
    )


def bootstrap_success_table(runs: list[RunBundle]) -> pd.DataFrame:
    rows = []
    for run in runs:
        tasks = tasks_frame(run)
        pass1 = [
            bool(record["attempts"] and record["attempts"][0]["passed"])
            for record in run.records
        ]
        bestn = tasks["solved"].astype(float)
        pass1_mean, pass1_low, pass1_high = bootstrap_ci(pass1)
        bestn_mean, bestn_low, bestn_high = bootstrap_ci(bestn)
        rows.extend(
            [
                {
                    "run": run.label,
                    "metric": "pass@1 estrito",
                    "mean": pass1_mean,
                    "ci_low": pass1_low,
                    "ci_high": pass1_high,
                },
                {
                    "run": run.label,
                    "metric": "Best-of-N observado",
                    "mean": bestn_mean,
                    "ci_low": bestn_low,
                    "ci_high": bestn_high,
                },
            ]
        )
    return pd.DataFrame(rows)


def _difficulty_label(correct_count: int, attempts: int) -> str:
    if correct_count == 0:
        return "unresolved"
    if correct_count == attempts:
        return "easy"
    if correct_count == 1:
        return "fragile"
    return "partial"
