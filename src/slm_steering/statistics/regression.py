from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from slm_steering.statistics.calibration import brier_score, log_loss
from slm_steering.statistics.permutation import roc_auc_score


@dataclass(frozen=True)
class LogisticModelResult:
    model_name: str
    auc: float
    brier_score: float
    log_loss: float
    accuracy: float
    n_train_tasks: int
    n_test_tasks: int
    n_train_samples: int
    n_test_samples: int
    coefficients: dict[str, float]
    odds_ratios: dict[str, float]

    def to_row(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "auc": self.auc,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "accuracy": self.accuracy,
            "n_train_tasks": self.n_train_tasks,
            "n_test_tasks": self.n_test_tasks,
            "n_train_samples": self.n_train_samples,
            "n_test_samples": self.n_test_samples,
            "coefficients_json": json.dumps(self.coefficients, sort_keys=True),
            "odds_ratios_json": json.dumps(self.odds_ratios, sort_keys=True),
        }


def build_success_regression_frame(
    latent_scores: pd.DataFrame,
    *,
    phase2_attempts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = latent_scores.copy()
    if "passed" not in frame.columns and "label" in frame.columns:
        frame["passed"] = frame["label"]
    if "attempt_index" not in frame.columns and "attempt" in frame.columns:
        frame["attempt_index"] = frame["attempt"]
    if "difficulty_class" not in frame.columns and "difficulty" in frame.columns:
        frame["difficulty_class"] = frame["difficulty"]

    if phase2_attempts is not None and not phase2_attempts.empty:
        merge_cols = [column for column in ["task_id", "attempt"] if column in frame.columns and column in phase2_attempts.columns]
        if merge_cols:
            extra_cols = [
                column
                for column in ["temperature", "top_p", "model_id", "generated_tokens", "generation_seconds"]
                if column in phase2_attempts.columns and column not in frame.columns
            ]
            frame = frame.merge(phase2_attempts[merge_cols + extra_cols], on=merge_cols, how="left")

    defaults = {
        "generated_tokens": 0.0,
        "generation_seconds": 0.0,
        "attempt_index": 1.0,
        "temperature": 0.0,
        "top_p": 1.0,
        "difficulty_class": "unknown",
        "model_id": "unknown",
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
    return frame


def compare_success_models(
    frame: pd.DataFrame,
    *,
    target: str = "passed",
    task_id_col: str = "task_id",
    seed: int = 1234,
    train_fraction: float = 0.7,
) -> pd.DataFrame:
    if frame.empty or target not in frame.columns or "latent_score" not in frame.columns:
        return empty_regression_results()

    observable_features = [
        column
        for column in [
            "generated_tokens",
            "generation_seconds",
            "attempt_index",
            "temperature",
            "top_p",
            "difficulty_class",
            "model_id",
        ]
        if column in frame.columns
    ]
    model_specs = [
        ("observables_only", observable_features),
        ("latent_only", ["latent_score"]),
        ("observables_plus_latent", observable_features + ["latent_score"]),
    ]
    rows = []
    for model_name, features in model_specs:
        result = fit_logistic_regression_model(
            frame,
            features=features,
            model_name=model_name,
            target=target,
            task_id_col=task_id_col,
            seed=seed,
            train_fraction=train_fraction,
        )
        if result is not None:
            rows.append(result.to_row())
    if not rows:
        return empty_regression_results()
    result_frame = pd.DataFrame(rows)
    if {"observables_only", "observables_plus_latent"}.issubset(set(result_frame["model_name"])):
        base_auc = float(result_frame.loc[result_frame["model_name"].eq("observables_only"), "auc"].iloc[0])
        result_frame["delta_auc_vs_observables"] = result_frame["auc"] - base_auc
    else:
        result_frame["delta_auc_vs_observables"] = np.nan
    return result_frame


def fit_logistic_regression_model(
    frame: pd.DataFrame,
    *,
    features: list[str],
    model_name: str = "logistic_regression",
    target: str = "passed",
    task_id_col: str = "task_id",
    seed: int = 1234,
    train_fraction: float = 0.7,
    steps: int = 400,
    lr: float = 0.1,
    l2: float = 1e-3,
) -> LogisticModelResult | None:
    required = [task_id_col, target, *features]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Colunas ausentes para regressao: {missing}")

    clean = frame[required].dropna(subset=[task_id_col, target]).copy()
    if clean.empty or clean[target].nunique() < 2:
        return None
    train_mask, test_mask = task_split_mask(
        clean[task_id_col],
        clean[target],
        train_fraction=train_fraction,
        seed=seed,
    )
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None
    y_train = clean.loc[train_mask, target].astype(int).to_numpy()
    y_test = clean.loc[test_mask, target].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    design = pd.get_dummies(clean[features], dummy_na=True, drop_first=False)
    design = design.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_train = design.loc[train_mask].to_numpy(dtype=np.float64)
    x_test = design.loc[test_mask].to_numpy(dtype=np.float64)
    x_train, x_test, mean, std = standardize_train_test(x_train, x_test)
    weights, bias = train_logistic_regression(x_train, y_train, steps=steps, lr=lr, l2=l2)
    probabilities = sigmoid(x_test @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)
    mean_flat = mean.flatten()
    std_flat = std.flatten()
    coefficients = {
        column: float(weight / std_flat[index])
        for index, (column, weight) in enumerate(zip(design.columns, weights, strict=False))
    }
    coefficients["intercept"] = float(bias - np.sum(weights * mean_flat / std_flat))
    odds_ratios = {key: float(np.exp(np.clip(value, -20, 20))) for key, value in coefficients.items()}
    return LogisticModelResult(
        model_name=model_name,
        auc=roc_auc_score(y_test, probabilities),
        brier_score=brier_score(probabilities, y_test),
        log_loss=log_loss(probabilities, y_test),
        accuracy=float((predictions == y_test).mean()),
        n_train_tasks=int(clean.loc[train_mask, task_id_col].nunique()),
        n_test_tasks=int(clean.loc[test_mask, task_id_col].nunique()),
        n_train_samples=int(train_mask.sum()),
        n_test_samples=int(test_mask.sum()),
        coefficients=coefficients,
        odds_ratios=odds_ratios,
    )


def task_split_mask(
    task_ids: pd.Series,
    labels: pd.Series,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[pd.Series, pd.Series]:
    task_frame = pd.DataFrame({"task_id": task_ids, "label": labels.astype(int)})
    task_labels = task_frame.groupby("task_id")["label"].max().reset_index()
    rng = np.random.default_rng(seed)
    train_tasks = []
    test_tasks = []
    for label in sorted(task_labels["label"].unique()):
        ids = task_labels.loc[task_labels["label"].eq(label), "task_id"].to_numpy(dtype=object)
        rng.shuffle(ids)
        if len(ids) == 1:
            train_tasks.extend(ids.tolist())
            continue
        split = max(1, int(round(len(ids) * train_fraction)))
        split = min(split, len(ids) - 1)
        train_tasks.extend(ids[:split].tolist())
        test_tasks.extend(ids[split:].tolist())
    if not test_tasks and train_tasks:
        test_tasks.append(train_tasks.pop())
    train_mask = task_ids.isin(train_tasks)
    test_mask = task_ids.isin(test_tasks)
    return train_mask, test_mask


def train_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    steps: int,
    lr: float,
    l2: float,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    y_float = y.astype(np.float64)
    for _ in range(steps):
        probabilities = sigmoid(x @ weights + bias)
        error = probabilities - y_float
        grad_w = (x.T @ error) / len(x) + l2 * weights
        grad_b = float(error.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights, bias


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -50, 50)))


def empty_regression_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model_name",
            "auc",
            "brier_score",
            "log_loss",
            "accuracy",
            "n_train_tasks",
            "n_test_tasks",
            "n_train_samples",
            "n_test_samples",
            "coefficients_json",
            "odds_ratios_json",
            "delta_auc_vs_observables",
        ]
    )
