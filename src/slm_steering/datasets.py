from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


DEFAULT_HUMANEVAL_DATASET = "openai/openai_humaneval"
DEFAULT_HUMANEVAL_PLUS_DATASET = "evalplus/humanevalplus"
DEFAULT_MBPP_DATASET = "google-research-datasets/mbpp"


@dataclass(frozen=True)
class CodingProblem:
    task_id: str
    prompt: str
    test: str
    entry_point: str
    benchmark: str | None = None
    difficulty: str | None = None
    source: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def load_humaneval(
    limit: int | None = None,
    offset: int = 0,
    dataset_id: str = DEFAULT_HUMANEVAL_DATASET,
    split: str = "test",
) -> list[CodingProblem]:
    """Load HumanEval problems from Hugging Face Datasets."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split)
    rows = _slice_rows(dataset, limit=limit, offset=offset)

    return [
        CodingProblem(
            task_id=row["task_id"],
            prompt=row["prompt"],
            test=row["test"],
            entry_point=row["entry_point"],
            benchmark="humaneval",
            source=dataset_id,
            tags=("code-generation", "python", "unit-tests"),
        )
        for row in rows
    ]


def load_humaneval_plus(
    limit: int | None = None,
    offset: int = 0,
    dataset_id: str = DEFAULT_HUMANEVAL_PLUS_DATASET,
    split: str = "test",
) -> list[CodingProblem]:
    """Load HumanEval+ style rows and normalize them to CodingProblem.

    EvalPlus mirrors the HumanEval task schema in common distributions, but
    some mirrors expose tests under slightly different column names. This
    adapter accepts the common variants and raises a clear error when a row
    cannot be represented as an executable check function.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split)
    rows = _slice_rows(dataset, limit=limit, offset=offset)

    problems = []
    for index, row in enumerate(rows):
        entry_point = _string_value(row, "entry_point", "function_name", "fn_name")
        if not entry_point:
            entry_point = _infer_entry_point(row)
        test = _coerce_test_function(row, entry_point=entry_point)
        problems.append(
            CodingProblem(
                task_id=str(row.get("task_id") or row.get("id") or f"HumanEvalPlus/{offset + index}"),
                prompt=str(row.get("prompt") or row.get("declaration") or ""),
                test=test,
                entry_point=entry_point,
                benchmark="humaneval_plus",
                source=dataset_id,
                tags=("code-generation", "python", "unit-tests", "evalplus"),
                metadata=_metadata(row, exclude={"prompt", "test", "entry_point"}),
            )
        )
    return problems


def load_mbpp(
    limit: int | None = None,
    offset: int = 0,
    dataset_id: str = DEFAULT_MBPP_DATASET,
    split: str = "test",
) -> list[CodingProblem]:
    """Load MBPP/MBPP Sanitized style rows as CodingProblem objects."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split)
    rows = _slice_rows(dataset, limit=limit, offset=offset)

    problems = []
    for index, row in enumerate(rows):
        entry_point = _string_value(row, "entry_point", "function_name", "fn_name")
        if not entry_point:
            entry_point = _infer_entry_point(row)
        task_text = _string_value(row, "prompt", "text", "description") or ""
        prompt = _mbpp_prompt(task_text, entry_point)
        setup = _string_value(row, "test_setup_code", "setup_code") or ""
        tests = _list_value(row, "test_list", "tests", "challenge_test_list")
        if not tests:
            raise ValueError(f"MBPP row sem testes executaveis: {row.get('task_id') or row.get('id') or index}")
        problems.append(
            CodingProblem(
                task_id=str(row.get("task_id") or row.get("id") or f"MBPP/{offset + index}"),
                prompt=prompt,
                test=_tests_to_check(tests, entry_point=entry_point, setup=setup),
                entry_point=entry_point,
                benchmark="mbpp",
                difficulty=str(row.get("difficulty")) if row.get("difficulty") is not None else None,
                source=dataset_id,
                tags=("code-generation", "python", "unit-tests", "mbpp"),
                metadata=_metadata(row, exclude={"prompt", "text", "test_list", "tests"}),
            )
        )
    return problems


def _slice_rows(rows: Iterable[dict[str, Any]], *, limit: int | None, offset: int) -> list[dict[str, Any]]:
    materialized = list(rows)
    start = max(0, offset)
    end = None if limit is None else start + limit
    return materialized[start:end]


def _string_value(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _list_value(row: dict[str, Any], *keys: str) -> list[str]:
    values = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, Iterable):
            values.extend(str(item).strip() for item in value if str(item).strip())
    return values


def _infer_entry_point(row: dict[str, Any]) -> str:
    candidates = []
    candidates.extend(_list_value(row, "test_list", "tests", "challenge_test_list"))
    for key in ["code", "canonical_solution", "prompt", "declaration", "text"]:
        value = row.get(key)
        if isinstance(value, str):
            candidates.append(value)
    text = "\n".join(candidates)
    for pattern in [
        r"\bassert\s+([A-Za-z_]\w*)\s*\(",
        r"\bdef\s+([A-Za-z_]\w*)\s*\(",
        r"function\s+named\s+[`'\"]?([A-Za-z_]\w*)",
    ]:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    raise ValueError(f"Nao foi possivel inferir entry_point para task {row.get('task_id') or row.get('id')}")


def _coerce_test_function(row: dict[str, Any], *, entry_point: str) -> str:
    direct = _string_value(row, "test", "canonical_tests")
    if direct:
        if "def check" in direct:
            return direct
        return _tests_to_check([direct], entry_point=entry_point)
    tests = _list_value(row, "base_tests", "plus_tests", "test_list", "tests")
    if tests:
        return _tests_to_check(tests, entry_point=entry_point)
    raise ValueError(f"HumanEval+ row sem testes executaveis: {row.get('task_id') or row.get('id')}")


def _tests_to_check(tests: list[str], *, entry_point: str, setup: str = "") -> str:
    lines = ["def check(candidate):"]
    if setup.strip():
        lines.extend(f"    {line}" if line.strip() else "" for line in setup.strip().splitlines())
    lines.append(f"    {entry_point} = candidate")
    for test in tests:
        for line in test.strip().splitlines():
            lines.append(f"    {line}" if line.strip() else "")
    return "\n".join(lines)


def _mbpp_prompt(task_text: str, entry_point: str) -> str:
    text = task_text.strip().replace("\r\n", "\n")
    return (
        f"# Task: {text}\n"
        f"# Write a Python function named {entry_point} that satisfies the tests.\n"
    )


def _metadata(row: dict[str, Any], *, exclude: set[str]) -> dict[str, Any]:
    metadata = {}
    for key, value in row.items():
        if key not in exclude and isinstance(value, (str, int, float, bool, type(None))):
            metadata[key] = value
    return metadata
