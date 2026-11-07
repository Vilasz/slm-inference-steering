from __future__ import annotations

from dataclasses import dataclass

from slm_steering.benchmarks.stress import load_humaneval_stress
from slm_steering.datasets import (
    DEFAULT_HUMANEVAL_DATASET,
    DEFAULT_HUMANEVAL_PLUS_DATASET,
    DEFAULT_MBPP_DATASET,
    CodingProblem,
    load_humaneval,
    load_humaneval_plus,
    load_mbpp,
)


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    name: str
    dataset_id: str
    split: str
    loader: str
    status: str
    notes: str


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "humaneval": BenchmarkSpec(
        key="humaneval",
        name="HumanEval",
        dataset_id=DEFAULT_HUMANEVAL_DATASET,
        split="test",
        loader="humaneval",
        status="implemented",
        notes="Canonical Python code-generation benchmark used by the baseline.",
    ),
    "humaneval_plus": BenchmarkSpec(
        key="humaneval_plus",
        name="HumanEval+",
        dataset_id=DEFAULT_HUMANEVAL_PLUS_DATASET,
        split="test",
        loader="humaneval_plus",
        status="implemented",
        notes="EvalPlus/HumanEval+ style adapter with schema fallbacks for stronger tests.",
    ),
    "humaneval_stress": BenchmarkSpec(
        key="humaneval_stress",
        name="HumanEval Stress",
        dataset_id="local/humaneval_stress",
        split="test",
        loader="humaneval_stress",
        status="implemented",
        notes=(
            "Small local stress suite with edge-case prompts derived from HumanEval-style "
            "tasks for robustness checks."
        ),
    ),
    "mbpp": BenchmarkSpec(
        key="mbpp",
        name="MBPP",
        dataset_id=DEFAULT_MBPP_DATASET,
        split="test",
        loader="mbpp",
        status="implemented",
        notes="MBPP/MBPP Sanitized style adapter that wraps assertion tests in check(candidate).",
    ),
    "ds1000": BenchmarkSpec(
        key="ds1000",
        name="DS-1000",
        dataset_id="xlangai/DS-1000",
        split="test",
        loader="not_implemented",
        status="planned",
        notes="Harder data-science benchmark; needs a separate verifier policy.",
    ),
}


def list_benchmarks() -> list[BenchmarkSpec]:
    return list(BENCHMARKS.values())


def get_benchmark(key: str) -> BenchmarkSpec:
    try:
        return BENCHMARKS[key]
    except KeyError as exc:
        available = ", ".join(sorted(BENCHMARKS))
        raise ValueError(f"Benchmark desconhecido: {key}. Opcoes: {available}") from exc


def load_benchmark(
    key: str,
    *,
    limit: int | None = None,
    offset: int = 0,
    dataset_id: str | None = None,
) -> list[CodingProblem]:
    spec = get_benchmark(key)
    resolved_dataset_id = dataset_id or spec.dataset_id
    if spec.loader == "humaneval":
        return load_humaneval(limit=limit, offset=offset, dataset_id=resolved_dataset_id, split=spec.split)
    if spec.loader == "humaneval_plus":
        return load_humaneval_plus(limit=limit, offset=offset, dataset_id=resolved_dataset_id, split=spec.split)
    if spec.loader == "humaneval_stress":
        return load_humaneval_stress(limit=limit, offset=offset)
    if spec.loader == "mbpp":
        return load_mbpp(limit=limit, offset=offset, dataset_id=resolved_dataset_id, split=spec.split)
    raise NotImplementedError(
        f"Benchmark '{key}' esta registrado, mas ainda nao tem loader implementado. "
        "Adicione um adaptador que retorne uma lista de CodingProblem."
    )
