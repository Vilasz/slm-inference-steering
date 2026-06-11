from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any


def extended_generation_metrics(records: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    attempts = [attempt for record in records for attempt in record.get("attempts", [])]
    total_attempts = len(attempts)
    errors = Counter(attempt.get("error_type") or "passed" for attempt in attempts)
    diversity = mean_task_diversity(records)
    solved = summary.get("observed_best_of_n") or 0.0
    mean_tokens = summary.get("mean_tokens_until_success_or_budget") or summary.get("mean_generated_tokens_per_attempt")
    return {
        "syntax_error_rate": errors.get("syntax_error", 0) / total_attempts if total_attempts else 0.0,
        "runtime_error_rate": errors.get("runtime_error", 0) / total_attempts if total_attempts else 0.0,
        "test_failure_rate": errors.get("test_failure", 0) / total_attempts if total_attempts else 0.0,
        "diversity_score": diversity,
        "collapse_score": 1 - diversity if diversity is not None else None,
        "accuracy_per_1k_tokens": (
            solved / (mean_tokens / 1000) if mean_tokens and mean_tokens > 0 else None
        ),
    }


def mean_task_diversity(records: list[dict[str, Any]]) -> float | None:
    values = []
    for record in records:
        sources = [attempt.get("candidate_source", "") or attempt.get("raw_completion", "") for attempt in record.get("attempts", [])]
        if len(sources) < 2:
            continue
        distances = [
            1 - SequenceMatcher(None, left, right).ratio()
            for left, right in combinations(sources, 2)
        ]
        if distances:
            values.append(sum(distances) / len(distances))
    if not values:
        return None
    return sum(values) / len(values)
