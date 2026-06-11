from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


MetricFn = Callable[[Any], float]


@dataclass(frozen=True)
class BootstrapSummary:
    metric_mean: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    confidence_level: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def stratified_task_bootstrap(
    data: pd.DataFrame | list[dict[str, Any]],
    metric_fn: MetricFn,
    *,
    task_id_col: str = "task_id",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 1234,
) -> dict[str, Any]:
    distribution = bootstrap_metric_distribution(
        data,
        metric_fn,
        task_id_col=task_id_col,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    summary = bootstrap_confidence_interval(
        distribution,
        confidence_level=confidence_level,
    )
    return {
        **summary.to_dict(),
        "distribution": distribution,
    }


def bootstrap_metric_distribution(
    data: pd.DataFrame | list[dict[str, Any]],
    metric_fn: MetricFn,
    *,
    task_id_col: str = "task_id",
    n_bootstrap: int = 1000,
    seed: int = 1234,
) -> np.ndarray:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap deve ser >= 1")
    groups = group_by_task(data, task_id_col=task_id_col)
    if not groups:
        return np.asarray([], dtype=np.float64)

    rng = np.random.default_rng(seed)
    task_ids = np.asarray(list(groups), dtype=object)
    values = np.zeros(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sampled_ids = rng.choice(task_ids, size=len(task_ids), replace=True)
        sample = concat_task_groups([groups[task_id] for task_id in sampled_ids], like=data)
        values[index] = float(metric_fn(sample))
    return values


def bootstrap_confidence_interval(
    distribution: np.ndarray | list[float],
    *,
    confidence_level: float = 0.95,
) -> BootstrapSummary:
    values = np.asarray(distribution, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return BootstrapSummary(
            metric_mean=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_bootstrap=0,
            confidence_level=confidence_level,
        )
    alpha = 1.0 - confidence_level
    return BootstrapSummary(
        metric_mean=float(values.mean()),
        ci_low=float(np.quantile(values, alpha / 2)),
        ci_high=float(np.quantile(values, 1 - alpha / 2)),
        n_bootstrap=int(values.size),
        confidence_level=confidence_level,
    )


def group_by_task(
    data: pd.DataFrame | list[dict[str, Any]],
    *,
    task_id_col: str = "task_id",
) -> dict[Any, Any]:
    if isinstance(data, pd.DataFrame):
        if task_id_col not in data.columns:
            raise KeyError(f"Coluna ausente: {task_id_col}")
        return {task_id: group.copy() for task_id, group in data.groupby(task_id_col, sort=False)}

    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in data:
        task_id = row.get(task_id_col)
        if task_id is None:
            raise KeyError(f"Registro sem {task_id_col}: {row}")
        groups.setdefault(task_id, []).append(row)
    return groups


def concat_task_groups(groups: list[Any], *, like: pd.DataFrame | list[dict[str, Any]]) -> Any:
    if isinstance(like, pd.DataFrame):
        if not groups:
            return like.iloc[0:0].copy()
        return pd.concat(groups, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for group in groups:
        rows.extend(group)
    return rows


def metric_pass_at_1(records: list[dict[str, Any]]) -> float:
    if not records:
        return float("nan")
    values = []
    for record in records:
        attempts = record.get("attempts", [])
        values.append(bool(attempts and attempts[0].get("passed")))
    return float(np.mean(values))


def metric_resolution_rate(records: list[dict[str, Any]]) -> float:
    if not records:
        return float("nan")
    return float(np.mean([bool(record.get("solved")) for record in records]))


def metric_best_of_k(records: list[dict[str, Any]], *, k: int) -> float:
    if not records:
        return float("nan")
    solved = []
    for record in records:
        first_success = record.get("first_success_attempt")
        solved.append(first_success is not None and first_success <= k)
    return float(np.mean(solved))


def metric_mean_tokens(records: list[dict[str, Any]]) -> float:
    totals = [
        sum(float(attempt.get("generated_tokens") or 0) for attempt in record.get("attempts", []))
        for record in records
    ]
    return float(np.mean(totals)) if totals else float("nan")


def metric_mean_latency(records: list[dict[str, Any]]) -> float:
    values = []
    for record in records:
        if record.get("problem_wall_seconds") is not None:
            values.append(float(record["problem_wall_seconds"]))
            continue
        values.append(
            sum(
                float(attempt.get("generation_seconds") or 0)
                + float(attempt.get("verification_seconds") or 0)
                for attempt in record.get("attempts", [])
            )
        )
    return float(np.mean(values)) if values else float("nan")


def metric_mean_attempts_until_success_or_budget(records: list[dict[str, Any]]) -> float:
    values = []
    for record in records:
        first_success = record.get("first_success_attempt")
        values.append(float(first_success if first_success is not None else len(record.get("attempts", []))))
    return float(np.mean(values)) if values else float("nan")
