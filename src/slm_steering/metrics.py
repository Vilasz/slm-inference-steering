from __future__ import annotations

from math import comb
from collections import Counter
from statistics import mean, median
from typing import Any


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k estimator used by the HumanEval paper."""
    if num_samples < k:
        raise ValueError("num_samples must be >= k")
    if num_correct == 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - comb(num_samples - num_correct, k) / comb(num_samples, k)


def summarize(records: list[dict[str, Any]], requested_n: int) -> dict[str, Any]:
    if not records:
        return {
            "num_tasks": 0,
            "requested_n": requested_n,
            "solved_tasks": 0,
        }

    all_attempts = [attempt for record in records for attempt in record["attempts"]]
    max_attempts = max(len(record["attempts"]) for record in records)
    adaptive_sampling_used = any(len(record["attempts"]) < requested_n for record in records)
    solved_records = [record for record in records if record["solved"]]
    first_attempt_correct = [
        bool(record["attempts"] and record["attempts"][0]["passed"])
        for record in records
    ]

    observed_best_of_k = {}
    for k in range(1, max_attempts + 1):
        values = []
        for record in records:
            first_success = record["first_success_attempt"]
            if first_success is not None:
                values.append(first_success <= k)
            elif len(record["attempts"]) >= k:
                values.append(False)
        if values:
            observed_best_of_k[str(k)] = sum(values) / len(values)

    pass_at_k_estimate = {}
    pass_at_k_eligible_tasks = {}
    for k in range(1, max_attempts + 1):
        values = []
        for record in records:
            attempts = record["attempts"]
            if len(attempts) < k:
                continue
            correct = sum(1 for attempt in attempts if attempt["passed"])
            values.append(estimate_pass_at_k(len(attempts), correct, k))
        if values:
            pass_at_k_estimate[str(k)] = mean(values)
            pass_at_k_eligible_tasks[str(k)] = len(values)

    first_success_attempts = [
        record["first_success_attempt"]
        for record in solved_records
        if record["first_success_attempt"] is not None
    ]
    attempts_until_success_or_budget = [
        record["first_success_attempt"]
        if record["first_success_attempt"] is not None
        else len(record["attempts"])
        for record in records
    ]

    generation_seconds = [attempt["generation_seconds"] for attempt in all_attempts]
    verification_seconds = [attempt["verification_seconds"] for attempt in all_attempts]
    generated_tokens = [attempt["generated_tokens"] for attempt in all_attempts]
    total_attempt_seconds = [
        attempt["generation_seconds"] + attempt["verification_seconds"]
        for attempt in all_attempts
    ]
    tokens_per_second = [
        attempt["generated_tokens"] / attempt["generation_seconds"]
        for attempt in all_attempts
        if attempt["generation_seconds"] > 0
    ]
    error_type_counts = Counter(
        attempt.get("error_type") or "passed" for attempt in all_attempts
    )

    tokens_until_success_or_budget = []
    seconds_until_success_or_budget = []
    effective_attempts_until_success_or_budget = []
    for record in records:
        cutoff = record["first_success_attempt"] or len(record["attempts"])
        considered_attempts = record["attempts"][:cutoff]
        effective_attempts_until_success_or_budget.append(cutoff)
        tokens_until_success_or_budget.append(
            sum(attempt["generated_tokens"] for attempt in considered_attempts)
        )
        seconds_until_success_or_budget.append(
            sum(
                attempt["generation_seconds"] + attempt["verification_seconds"]
                for attempt in considered_attempts
            )
        )

    total_effective_tokens = sum(tokens_until_success_or_budget)
    total_effective_seconds = sum(seconds_until_success_or_budget)
    total_generated_tokens = sum(generated_tokens)
    total_wall_seconds = sum(total_attempt_seconds)

    return {
        "num_tasks": len(records),
        "requested_n": requested_n,
        "adaptive_sampling_used": adaptive_sampling_used,
        "total_attempts": len(all_attempts),
        "solved_tasks": len(solved_records),
        "strict_pass_at_1": sum(first_attempt_correct) / len(first_attempt_correct),
        "observed_best_of_n": len(solved_records) / len(records),
        "observed_best_of_k": observed_best_of_k,
        "pass_at_k_estimate": pass_at_k_estimate,
        "pass_at_k_eligible_tasks": pass_at_k_eligible_tasks,
        "mean_generated_tokens_per_attempt": mean(generated_tokens),
        "median_generated_tokens_per_attempt": median(generated_tokens),
        "total_generated_tokens": total_generated_tokens,
        "mean_generation_seconds_per_attempt": mean(generation_seconds),
        "total_generation_seconds": sum(generation_seconds),
        "mean_verification_seconds_per_attempt": mean(verification_seconds),
        "total_verification_seconds": sum(verification_seconds),
        "mean_total_seconds_per_attempt": mean(total_attempt_seconds),
        "mean_generation_tokens_per_second": mean(tokens_per_second) if tokens_per_second else None,
        "mean_problem_wall_seconds": mean(record["problem_wall_seconds"] for record in records),
        "mean_attempts_until_first_success_solved_only": (
            mean(first_success_attempts) if first_success_attempts else None
        ),
        "median_attempts_until_first_success_solved_only": (
            median(first_success_attempts) if first_success_attempts else None
        ),
        "mean_attempts_until_success_or_budget": mean(attempts_until_success_or_budget),
        "median_attempts_until_success_or_budget": median(attempts_until_success_or_budget),
        "total_effective_attempts_until_success_or_budget": sum(
            effective_attempts_until_success_or_budget
        ),
        "mean_tokens_until_success_or_budget": mean(tokens_until_success_or_budget),
        "total_tokens_until_success_or_budget": total_effective_tokens,
        "mean_seconds_until_success_or_budget": mean(seconds_until_success_or_budget),
        "total_seconds_until_success_or_budget": total_effective_seconds,
        "token_savings_if_oracle_early_stop": (
            1 - total_effective_tokens / total_generated_tokens
            if total_generated_tokens
            else None
        ),
        "time_savings_if_oracle_early_stop": (
            1 - total_effective_seconds / total_wall_seconds
            if total_wall_seconds
            else None
        ),
        "generated_tokens_per_solved_task": (
            total_generated_tokens / len(solved_records) if solved_records else None
        ),
        "effective_tokens_per_solved_task": (
            total_effective_tokens / len(solved_records) if solved_records else None
        ),
        "error_type_counts": dict(sorted(error_type_counts.items())),
    }
