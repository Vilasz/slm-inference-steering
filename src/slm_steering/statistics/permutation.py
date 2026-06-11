from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PermutationTestResult:
    observed_score: float
    null_mean: float
    null_std: float
    p_value: float
    n_permutations: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def permutation_test_auc(
    scores: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    *,
    groups: list[Any] | np.ndarray | None = None,
    n_permutations: int = 1000,
    seed: int = 1234,
) -> PermutationTestResult:
    score_values = np.asarray(scores, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int64)
    observed = roc_auc_score(label_values, score_values)
    null = _permutation_null(
        score_values,
        label_values,
        lambda y: roc_auc_score(y, score_values),
        groups=groups,
        n_permutations=n_permutations,
        seed=seed,
    )
    return summarize_null(observed, null)


def permutation_test_accuracy(
    scores: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    *,
    threshold: float = 0.5,
    groups: list[Any] | np.ndarray | None = None,
    n_permutations: int = 1000,
    seed: int = 1234,
) -> PermutationTestResult:
    score_values = np.asarray(scores, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int64)
    predicted = (score_values >= threshold).astype(np.int64)
    observed = accuracy_score(label_values, predicted)
    null = _permutation_null(
        predicted,
        label_values,
        lambda y: accuracy_score(y, predicted),
        groups=groups,
        n_permutations=n_permutations,
        seed=seed,
    )
    return summarize_null(observed, null)


def shuffle_labels_within_groups(
    labels: list[int] | np.ndarray,
    *,
    groups: list[Any] | np.ndarray | pd.DataFrame | None = None,
    group_cols: list[str] | tuple[str, ...] | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    labels_array = np.asarray(labels).copy()
    random = rng or np.random.default_rng(seed)
    if groups is None:
        random.shuffle(labels_array)
        return labels_array

    group_keys = normalize_group_keys(groups, group_cols=group_cols)
    if len(group_keys) != len(labels_array):
        raise ValueError("groups e labels devem ter o mesmo tamanho")
    for key in pd.Series(group_keys).drop_duplicates().tolist():
        indices = np.where(group_keys == key)[0]
        labels_array[indices] = random.permutation(labels_array[indices])
    return labels_array


def normalize_group_keys(
    groups: list[Any] | np.ndarray | pd.DataFrame,
    *,
    group_cols: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    if isinstance(groups, pd.DataFrame):
        if not group_cols:
            raise ValueError("group_cols e obrigatorio quando groups e DataFrame")
        missing = [column for column in group_cols if column not in groups.columns]
        if missing:
            raise KeyError(f"Colunas de grupo ausentes: {missing}")
        return groups.loc[:, list(group_cols)].astype(str).agg("||".join, axis=1).to_numpy(dtype=object)
    return np.asarray(groups, dtype=object)


def summarize_null(observed: float, null: np.ndarray) -> PermutationTestResult:
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return PermutationTestResult(
            observed_score=float(observed),
            null_mean=float("nan"),
            null_std=float("nan"),
            p_value=float("nan"),
            n_permutations=0,
        )
    extreme = np.abs(null - null.mean()) >= abs(observed - null.mean())
    p_value = (int(extreme.sum()) + 1) / (len(null) + 1)
    return PermutationTestResult(
        observed_score=float(observed),
        null_mean=float(null.mean()),
        null_std=float(null.std(ddof=1)) if len(null) > 1 else 0.0,
        p_value=float(p_value),
        n_permutations=int(len(null)),
    )


def _permutation_null(
    _scores: np.ndarray,
    labels: np.ndarray,
    score_fn,
    *,
    groups: list[Any] | np.ndarray | None,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_permutations):
        shuffled = shuffle_labels_within_groups(labels, groups=groups, rng=rng)
        values.append(float(score_fn(shuffled)))
    return np.asarray(values, dtype=np.float64)


def accuracy_score(labels: np.ndarray, predictions: np.ndarray) -> float:
    if len(labels) == 0:
        return float("nan")
    return float((labels.astype(int) == predictions.astype(int)).mean())


def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    greater = 0.0
    for value in positives:
        greater += float((value > negatives).sum())
        greater += 0.5 * float((value == negatives).sum())
    return float(greater / (len(positives) * len(negatives)))
