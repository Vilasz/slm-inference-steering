from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DifficultyConfig:
    easy_success_rate: float = 0.6
    fragile_success_rate: float = 0.25


def classify_task_difficulty(
    record: dict[str, Any],
    config: DifficultyConfig | None = None,
) -> dict[str, Any]:
    cfg = config or DifficultyConfig()
    attempts = record.get("attempts", [])
    total_attempts = len(attempts)
    correct_count = sum(1 for attempt in attempts if attempt.get("passed"))
    success_rate = correct_count / total_attempts if total_attempts else 0.0
    first_success_attempt = record.get("first_success_attempt")
    pass_at_1 = bool(attempts and attempts[0].get("passed"))
    solved_by_sampling = bool(not pass_at_1 and first_success_attempt is not None)

    if correct_count == 0:
        difficulty = "hard"
    elif pass_at_1 or success_rate >= cfg.easy_success_rate:
        difficulty = "easy"
    elif success_rate <= cfg.fragile_success_rate:
        difficulty = "fragile"
    else:
        difficulty = "sampling_sensitive"

    return {
        "task_id": record.get("task_id"),
        "entry_point": record.get("entry_point"),
        "difficulty": difficulty,
        "attempts": total_attempts,
        "correct_count": correct_count,
        "success_rate": success_rate,
        "first_success_attempt": first_success_attempt,
        "pass_at_1": pass_at_1,
        "solved_by_sampling": solved_by_sampling,
        "generated_tokens": sum(attempt.get("generated_tokens", 0) for attempt in attempts),
        "generation_seconds": sum(attempt.get("generation_seconds", 0.0) for attempt in attempts),
        "verification_seconds": sum(attempt.get("verification_seconds", 0.0) for attempt in attempts),
    }


def difficulty_frame(
    records: list[dict[str, Any]],
    *,
    run_label: str | None = None,
    config: DifficultyConfig | None = None,
) -> pd.DataFrame:
    rows = []
    for record in records:
        row = classify_task_difficulty(record, config=config)
        if run_label is not None:
            row["run"] = run_label
        rows.append(row)
    return pd.DataFrame(rows)


def difficulty_distribution(
    records: list[dict[str, Any]],
    *,
    run_label: str | None = None,
    config: DifficultyConfig | None = None,
) -> pd.DataFrame:
    frame = difficulty_frame(records, run_label=run_label, config=config)
    if frame.empty:
        return pd.DataFrame(columns=["run", "difficulty", "tasks", "share"])
    group_cols = ["difficulty"]
    if run_label is not None:
        group_cols.insert(0, "run")
    counts = frame.groupby(group_cols, dropna=False).size().reset_index(name="tasks")
    total = counts["tasks"].sum() if run_label is None else counts.groupby("run")["tasks"].transform("sum")
    counts["share"] = counts["tasks"] / total
    return counts
