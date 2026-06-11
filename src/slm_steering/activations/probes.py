from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from slm_steering.activations.storage import ActivationDataset
from slm_steering.statistics.calibration import brier_score, expected_calibration_error
from slm_steering.statistics.permutation import roc_auc_score
from slm_steering.statistics.regression import sigmoid, standardize_train_test, task_split_mask, train_logistic_regression


@dataclass(frozen=True)
class ProbeEvaluationConfig:
    probe_types: tuple[str, ...] = ("logistic_regression", "ridge_classifier")
    train_fraction: float = 0.7
    seed: int = 1234
    steps: int = 400
    lr: float = 0.1
    l2: float = 1e-3


def evaluate_layerwise_probes(
    dataset: ActivationDataset,
    *,
    config: ProbeEvaluationConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ProbeEvaluationConfig()
    metadata = dataset.metadata.reset_index(drop=True).copy()
    if "task_id" not in metadata.columns:
        metadata["task_id"] = [f"sample/{index}" for index in range(dataset.num_samples)]
    rows = []
    for layer_position, layer in enumerate(dataset.layers):
        x = dataset.activations[:, layer_position, :].astype(np.float32)
        for probe_type in cfg.probe_types:
            rows.append(
                evaluate_probe(
                    x,
                    dataset.labels,
                    metadata=metadata,
                    layer=layer,
                    probe_type=probe_type,
                    config=cfg,
                )
            )
    return pd.DataFrame(rows)


def evaluate_probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    metadata: pd.DataFrame,
    layer: int,
    probe_type: str,
    config: ProbeEvaluationConfig,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    base = {
        "layer": layer,
        "position": _position_label(metadata),
        "probe_type": probe_type,
        "auc": np.nan,
        "accuracy": np.nan,
        "f1": np.nan,
        "brier_score": np.nan,
        "calibration_error": np.nan,
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
        "n_train_tasks": 0,
        "n_test_tasks": 0,
        "n_train_samples": 0,
        "n_test_samples": 0,
    }
    if len(x) < 4 or len(np.unique(y)) < 2:
        return base

    train_mask, test_mask = task_split_mask(
        metadata["task_id"],
        pd.Series(y),
        train_fraction=config.train_fraction,
        seed=config.seed,
    )
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return base
    y_train = y[train_mask.to_numpy()]
    y_test = y[test_mask.to_numpy()]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return base

    x_train = x[train_mask.to_numpy()]
    x_test = x[test_mask.to_numpy()]
    x_train, x_test, _, _ = standardize_train_test(x_train, x_test)
    if probe_type == "logistic_regression":
        scores = _logistic_scores(x_train, y_train, x_test, config=config)
    elif probe_type == "ridge_classifier":
        scores = _ridge_scores(x_train, y_train, x_test, l2=config.l2)
    else:
        raise ValueError(f"probe_type desconhecido: {probe_type}")

    probabilities = sigmoid(scores)
    predictions = (probabilities >= 0.5).astype(np.int64)
    confusion = confusion_counts(y_test, predictions)
    base.update(
        {
            "auc": roc_auc_score(y_test, probabilities),
            "accuracy": float((predictions == y_test).mean()),
            "f1": f1_score(y_test, predictions),
            "brier_score": brier_score(probabilities, y_test),
            "calibration_error": expected_calibration_error(probabilities, y_test, n_bins=5),
            "true_positive": confusion["true_positive"],
            "false_positive": confusion["false_positive"],
            "true_negative": confusion["true_negative"],
            "false_negative": confusion["false_negative"],
            "n_train_tasks": int(metadata.loc[train_mask, "task_id"].nunique()),
            "n_test_tasks": int(metadata.loc[test_mask, "task_id"].nunique()),
            "n_train_samples": int(train_mask.sum()),
            "n_test_samples": int(test_mask.sum()),
        }
    )
    return base


def _logistic_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    config: ProbeEvaluationConfig,
) -> np.ndarray:
    weights, bias = train_logistic_regression(
        x_train,
        y_train,
        steps=config.steps,
        lr=config.lr,
        l2=config.l2,
    )
    return x_test @ weights + bias


def _ridge_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    l2: float,
) -> np.ndarray:
    y_signed = np.where(y_train.astype(int) == 1, 1.0, -1.0)
    x_aug = np.column_stack([x_train, np.ones(len(x_train))])
    penalty = np.eye(x_aug.shape[1]) * l2
    penalty[-1, -1] = 0.0
    weights = np.linalg.pinv(x_aug.T @ x_aug + penalty) @ x_aug.T @ y_signed
    return np.column_stack([x_test, np.ones(len(x_test))]) @ weights


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    counts = confusion_counts(y_true, y_pred)
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    return {
        "true_positive": int(((y_true == 1) & (y_pred == 1)).sum()),
        "false_positive": int(((y_true == 0) & (y_pred == 1)).sum()),
        "true_negative": int(((y_true == 0) & (y_pred == 0)).sum()),
        "false_negative": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def _position_label(metadata: pd.DataFrame) -> str:
    if "token_position" not in metadata.columns:
        return "unknown"
    values = metadata["token_position"].dropna().astype(str).unique()
    if len(values) == 1:
        return values[0]
    if len(values) == 0:
        return "unknown"
    return "mixed"
