from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from typing import Any

import numpy as np
import pandas as pd

from slm_steering.statistics.bootstrap import bootstrap_confidence_interval


@dataclass(frozen=True)
class PairedTestResult:
    mean_difference: float
    median_difference: float
    ci_low: float
    ci_high: float
    p_value_if_available: float | None
    effect_size: float
    n_tasks: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def paired_task_difference(
    left: list[dict[str, Any]] | pd.DataFrame,
    right: list[dict[str, Any]] | pd.DataFrame,
    *,
    metric: str = "success",
    task_id_col: str = "task_id",
) -> pd.DataFrame:
    left_metrics = task_metric_frame(left, metric=metric, task_id_col=task_id_col, value_name="left")
    right_metrics = task_metric_frame(right, metric=metric, task_id_col=task_id_col, value_name="right")
    paired = left_metrics.merge(right_metrics, on=task_id_col, how="inner")
    paired["difference"] = paired["right"] - paired["left"]
    return paired


def paired_bootstrap_test(
    left: list[dict[str, Any]] | pd.DataFrame,
    right: list[dict[str, Any]] | pd.DataFrame,
    *,
    metric: str = "success",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 1234,
    task_id_col: str = "task_id",
) -> PairedTestResult:
    paired = paired_task_difference(left, right, metric=metric, task_id_col=task_id_col)
    differences = paired["difference"].to_numpy(dtype=np.float64)
    if differences.size == 0:
        return PairedTestResult(
            mean_difference=float("nan"),
            median_difference=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            p_value_if_available=None,
            effect_size=float("nan"),
            n_tasks=0,
        )
    rng = np.random.default_rng(seed)
    distribution = np.asarray(
        [
            float(rng.choice(differences, size=len(differences), replace=True).mean())
            for _ in range(n_bootstrap)
        ],
        dtype=np.float64,
    )
    ci = bootstrap_confidence_interval(distribution, confidence_level=confidence_level)
    sign = wilcoxon_or_sign_test(differences)
    return PairedTestResult(
        mean_difference=float(differences.mean()),
        median_difference=float(np.median(differences)),
        ci_low=ci.ci_low,
        ci_high=ci.ci_high,
        p_value_if_available=sign["p_value_if_available"],
        effect_size=standardized_effect_size(differences),
        n_tasks=int(differences.size),
    )


def wilcoxon_or_sign_test(differences: list[float] | np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    non_zero = values[values != 0]
    n = int(non_zero.size)
    if n == 0:
        return {"p_value_if_available": 1.0, "n_nonzero": 0, "positive": 0, "negative": 0}

    positive = int((non_zero > 0).sum())
    negative = int((non_zero < 0).sum())
    smaller = min(positive, negative)
    tail = sum(comb(n, k) for k in range(smaller + 1)) / (2**n)
    p_value = min(1.0, 2 * tail)
    return {
        "p_value_if_available": float(p_value),
        "n_nonzero": n,
        "positive": positive,
        "negative": negative,
    }


def task_metric_frame(
    data: list[dict[str, Any]] | pd.DataFrame,
    *,
    metric: str,
    task_id_col: str = "task_id",
    value_name: str = "value",
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        if task_id_col not in data.columns:
            raise KeyError(f"Coluna ausente: {task_id_col}")
        if metric in data.columns:
            frame = data[[task_id_col, metric]].copy()
            return frame.groupby(task_id_col, as_index=False)[metric].mean().rename(columns={metric: value_name})
        if metric == "success" and "passed" in data.columns:
            frame = data.groupby(task_id_col, as_index=False)["passed"].max()
            return frame.rename(columns={"passed": value_name})
        if metric == "tokens" and "generated_tokens" in data.columns:
            frame = data.groupby(task_id_col, as_index=False)["generated_tokens"].sum()
            return frame.rename(columns={"generated_tokens": value_name})
        raise ValueError(f"Metrica nao suportada para DataFrame: {metric}")

    rows = []
    for record in data:
        rows.append({task_id_col: record[task_id_col], value_name: record_metric(record, metric)})
    return pd.DataFrame(rows)


def record_metric(record: dict[str, Any], metric: str) -> float:
    attempts = record.get("attempts", [])
    if metric in {"success", "solved"}:
        return float(bool(record.get("solved")))
    if metric == "tokens":
        return float(sum(float(attempt.get("generated_tokens") or 0) for attempt in attempts))
    if metric == "latency":
        if record.get("problem_wall_seconds") is not None:
            return float(record["problem_wall_seconds"])
        return float(
            sum(
                float(attempt.get("generation_seconds") or 0)
                + float(attempt.get("verification_seconds") or 0)
                for attempt in attempts
            )
        )
    if metric == "first_success_attempt":
        first_success = record.get("first_success_attempt")
        return float(first_success if first_success is not None else len(attempts) + 1)
    if metric == "accuracy_per_1k_tokens":
        tokens = record_metric(record, "tokens")
        return float(record_metric(record, "success") / (tokens / 1000)) if tokens > 0 else 0.0
    raise ValueError(f"Metrica pareada desconhecida: {metric}")


def standardized_effect_size(differences: np.ndarray) -> float:
    if differences.size == 0:
        return float("nan")
    std = float(differences.std(ddof=1)) if differences.size > 1 else 0.0
    if std == 0:
        return 0.0
    return float(differences.mean() / std)
