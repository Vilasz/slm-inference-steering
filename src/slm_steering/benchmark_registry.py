from __future__ import annotations

from dataclasses import dataclass

from slm_steering.benchmarks.stress import load_humaneval_stress
from slm_steering.datasets import DEFAULT_HUMANEVAL_DATASET, CodingProblem, load_humaneval


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
        dataset_id="evalplus/humanevalplus",
        split="test",
        loader="not_implemented",
        status="planned",
        notes="EvalPlus adds stronger tests; adapter is planned because schemas differ.",
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
        dataset_id="google-research-datasets/mbpp",
        split="test",
        loader="not_implemented",
        status="planned",
        notes="Requires translating MBPP prompts/tests into the CodingProblem schema.",
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
        return load_humaneval(limit=limit, offset=offset, dataset_id=resolved_dataset_id)
    if spec.loader == "humaneval_stress":
        return load_humaneval_stress(limit=limit, offset=offset)
    raise NotImplementedError(
        f"Benchmark '{key}' esta registrado, mas ainda nao tem loader implementado. "
        "Adicione um adaptador que retorne uma lista de CodingProblem."
    )
