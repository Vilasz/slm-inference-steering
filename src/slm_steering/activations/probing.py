from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from slm_steering.activations.storage import ActivationDataset


@dataclass(frozen=True)
class ProbeResult:
    layer: int
    n_train: int
    n_test: int
    train_accuracy: float
    test_accuracy: float
    weights: np.ndarray
    bias: float


def latent_directions(dataset: ActivationDataset) -> dict[int, np.ndarray]:
    directions = {}
    for layer_position, layer in enumerate(dataset.layers):
        x = dataset.activations[:, layer_position, :]
        y = dataset.labels
        if len(np.unique(y)) < 2:
            continue
        correct = x[y == 1]
        incorrect = x[y == 0]
        directions[layer] = correct.mean(axis=0) - incorrect.mean(axis=0)
    return directions


def centroid_separability_frame(dataset: ActivationDataset) -> pd.DataFrame:
    rows = []
    for layer_position, layer in enumerate(dataset.layers):
        x = dataset.activations[:, layer_position, :]
        y = dataset.labels
        correct = x[y == 1]
        incorrect = x[y == 0]
        if len(correct) == 0 or len(incorrect) == 0:
            rows.append(
                {
                    "layer": layer,
                    "n_correct": len(correct),
                    "n_incorrect": len(incorrect),
                    "centroid_distance": np.nan,
                    "centroid_cosine": np.nan,
                    "fisher_ratio": np.nan,
                }
            )
            continue
        mean_correct = correct.mean(axis=0)
        mean_incorrect = incorrect.mean(axis=0)
        direction = mean_correct - mean_incorrect
        within = correct.var(axis=0).mean() + incorrect.var(axis=0).mean()
        rows.append(
            {
                "layer": layer,
                "n_correct": len(correct),
                "n_incorrect": len(incorrect),
                "centroid_distance": float(np.linalg.norm(direction)),
                "centroid_cosine": _cosine(mean_correct, mean_incorrect),
                "fisher_ratio": float(np.dot(direction, direction) / within) if within > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def probe_layers(
    dataset: ActivationDataset,
    *,
    train_fraction: float = 0.7,
    seed: int = 1234,
    steps: int = 300,
    lr: float = 0.1,
    l2: float = 1e-3,
) -> pd.DataFrame:
    rows = []
    for layer_position, layer in enumerate(dataset.layers):
        result = train_linear_probe(
            dataset.activations[:, layer_position, :],
            dataset.labels,
            layer=layer,
            train_fraction=train_fraction,
            seed=seed,
            steps=steps,
            lr=lr,
            l2=l2,
        )
        if result is None:
            rows.append(
                {
                    "layer": layer,
                    "n_train": 0,
                    "n_test": 0,
                    "train_accuracy": np.nan,
                    "test_accuracy": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "layer": result.layer,
                    "n_train": result.n_train,
                    "n_test": result.n_test,
                    "train_accuracy": result.train_accuracy,
                    "test_accuracy": result.test_accuracy,
                }
            )
    return pd.DataFrame(rows)


def train_linear_probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    layer: int = -1,
    train_fraction: float = 0.7,
    seed: int = 1234,
    steps: int = 300,
    lr: float = 0.1,
    l2: float = 1e-3,
) -> ProbeResult | None:
    if len(x) < 4 or len(np.unique(y)) < 2:
        return None
    rng = np.random.default_rng(seed)
    train_idx, test_idx = _stratified_split(y, train_fraction=train_fraction, rng=rng)
    if len(train_idx) == 0 or len(test_idx) == 0 or len(np.unique(y[train_idx])) < 2:
        return None

    x_train = x[train_idx].astype(np.float32)
    x_test = x[test_idx].astype(np.float32)
    y_train = y[train_idx].astype(np.float32)
    y_test = y[test_idx].astype(np.float32)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    weights = np.zeros(x_train.shape[1], dtype=np.float32)
    bias = 0.0
    for _ in range(steps):
        logits = x_train @ weights + bias
        probs = _sigmoid(logits)
        error = probs - y_train
        grad_w = (x_train.T @ error) / len(x_train) + l2 * weights
        grad_b = float(error.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b

    train_accuracy = _accuracy(x_train @ weights + bias, y_train)
    test_accuracy = _accuracy(x_test @ weights + bias, y_test)
    return ProbeResult(
        layer=layer,
        n_train=len(train_idx),
        n_test=len(test_idx),
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        weights=weights,
        bias=bias,
    )


def pca_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("pca_2d espera matriz [samples, features]")
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return centered @ components


def best_layer_candidate(separability: pd.DataFrame, probes: pd.DataFrame | None = None) -> int | None:
    frame = separability.copy()
    if probes is not None and not probes.empty:
        frame = frame.merge(probes[["layer", "test_accuracy"]], on="layer", how="left")
        frame["score"] = frame["test_accuracy"].fillna(0) + frame["fisher_ratio"].fillna(0)
    else:
        frame["score"] = frame["fisher_ratio"].fillna(0)
    if frame.empty:
        return None
    return int(frame.sort_values("score", ascending=False).iloc[0]["layer"])


def _stratified_split(y: np.ndarray, *, train_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train = []
    test = []
    for label in np.unique(y):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        split = max(1, int(round(len(indices) * train_fraction)))
        split = min(split, len(indices) - 1) if len(indices) > 1 else len(indices)
        train.extend(indices[:split])
        test.extend(indices[split:])
    rng.shuffle(train)
    rng.shuffle(test)
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


def _accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float(((logits >= 0).astype(np.float32) == y).mean())


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return np.nan
    return float(np.dot(left, right) / denom)
