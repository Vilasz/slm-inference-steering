from __future__ import annotations

from typing import Any

import pandas as pd


def best_of_k_marginal_curve(
    records: list[dict[str, Any]],
    *,
    max_k: int | None = None,
    stop_at_success: bool = False,
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "k",
                "eligible_tasks",
                "best_of_k_accuracy",
                "marginal_gain_k",
                "total_tokens_until_k",
                "mean_tokens_until_k",
                "total_seconds_until_k",
                "mean_seconds_until_k",
                "accuracy_per_1k_tokens",
            ]
        )

    budget = max_k or max(len(record.get("attempts", [])) for record in records)
    rows = []
    previous_accuracy = 0.0
    for k in range(1, budget + 1):
        outcomes = []
        token_costs = []
        second_costs = []
        for record in records:
            attempts = record.get("attempts", [])
            if not attempts:
                continue
            first_success = record.get("first_success_attempt")
            solved_by_k = first_success is not None and first_success <= k
            if len(attempts) < k and not solved_by_k:
                continue

            cutoff = min(k, len(attempts))
            if stop_at_success and solved_by_k:
                cutoff = int(first_success)
            considered = attempts[:cutoff]
            outcomes.append(solved_by_k)
            token_costs.append(sum(attempt.get("generated_tokens", 0) for attempt in considered))
            second_costs.append(
                sum(
                    attempt.get("generation_seconds", 0.0)
                    + attempt.get("verification_seconds", 0.0)
                    for attempt in considered
                )
            )

        if not outcomes:
            continue
        accuracy = sum(outcomes) / len(outcomes)
        total_tokens = sum(token_costs)
        mean_tokens = total_tokens / len(token_costs) if token_costs else 0.0
        total_seconds = sum(second_costs)
        mean_seconds = total_seconds / len(second_costs) if second_costs else 0.0
        rows.append(
            {
                "k": k,
                "eligible_tasks": len(outcomes),
                "best_of_k_accuracy": accuracy,
                "marginal_gain_k": accuracy - previous_accuracy,
                "total_tokens_until_k": total_tokens,
                "mean_tokens_until_k": mean_tokens,
                "total_seconds_until_k": total_seconds,
                "mean_seconds_until_k": mean_seconds,
                "accuracy_per_1k_tokens": (
                    accuracy / (mean_tokens / 1000) if mean_tokens > 0 else None
                ),
            }
        )
        previous_accuracy = accuracy
    return pd.DataFrame(rows)


def solved_by_sampling_tasks(records: list[dict[str, Any]]) -> list[str]:
    task_ids = []
    for record in records:
        attempts = record.get("attempts", [])
        pass_at_1 = bool(attempts and attempts[0].get("passed"))
        if not pass_at_1 and record.get("first_success_attempt") is not None:
            task_ids.append(record.get("task_id"))
    return task_ids
