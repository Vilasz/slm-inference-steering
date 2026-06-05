from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DEFAULT_HUMANEVAL_DATASET = "openai/openai_humaneval"


@dataclass(frozen=True)
class CodingProblem:
    task_id: str
    prompt: str
    test: str
    entry_point: str


def load_humaneval(
    limit: int | None = None,
    offset: int = 0,
    dataset_id: str = DEFAULT_HUMANEVAL_DATASET,
) -> list[CodingProblem]:
    """Load HumanEval problems from Hugging Face Datasets."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split="test")
    rows: Iterable[dict] = dataset
    if offset:
        rows = list(rows)[offset:]
    if limit is not None:
        rows = list(rows)[:limit]

    return [
        CodingProblem(
            task_id=row["task_id"],
            prompt=row["prompt"],
            test=row["test"],
            entry_point=row["entry_point"],
        )
        for row in rows
    ]
