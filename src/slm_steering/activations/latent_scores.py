from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from slm_steering.activations.storage import ActivationDataset
from slm_steering.statistics.permutation import roc_auc_score, shuffle_labels_within_groups


def compute_correctness_direction(
    activations: np.ndarray,
    labels: np.ndarray,
    *,
    layers: list[int] | None = None,
) -> dict[int, np.ndarray]:
    if activations.ndim != 3:
        raise ValueError("activations deve ter shape [samples, layers, hidden_size]")
    labels = np.asarray(labels, dtype=np.int64)
    if activations.shape[0] != labels.shape[0]:
        raise ValueError("activations e labels devem ter o mesmo numero de amostras")
    resolved_layers = layers or list(range(activations.shape[1]))
    directions = {}
    for layer_position, layer in enumerate(resolved_layers):
        x = activations[:, layer_position, :].astype(np.float32)
        if len(np.unique(labels)) < 2:
            continue
        correct = x[labels == 1]
        incorrect = x[labels == 0]
        if len(correct) == 0 or len(incorrect) == 0:
            continue
        directions[int(layer)] = normalize(correct.mean(axis=0) - incorrect.mean(axis=0))
    return directions


def project_onto_direction(
    activation: np.ndarray,
    direction: np.ndarray,
    *,
    normalize_activation: bool = True,
    normalize_direction: bool = True,
) -> np.ndarray:
    x = np.asarray(activation, dtype=np.float32)
    vector = np.asarray(direction, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError("direction deve ser vetor 1D")
    if x.shape[-1] != vector.shape[0]:
        raise ValueError(f"hidden_size incompativel: {x.shape[-1]} != {vector.shape[0]}")
    if normalize_activation:
        x = normalize_rows(x)
    if normalize_direction:
        vector = normalize(vector)
    return x @ vector


def compute_layerwise_latent_scores(
    dataset: ActivationDataset,
    *,
    direction_type: str = "correctness_direction",
    seed: int = 1234,
) -> pd.DataFrame:
    directions = build_direction_control(dataset, direction_type=direction_type, seed=seed)
    rows = []
    metadata = dataset.metadata.reset_index(drop=True)
    for layer_position, layer in enumerate(dataset.layers):
        direction = directions.get(layer)
        if direction is None:
            continue
        scores = project_onto_direction(dataset.activations[:, layer_position, :], direction)
        auc = roc_auc_score(dataset.labels, scores)
        for sample_index, score in enumerate(scores):
            row: dict[str, Any] = {
                "sample_index": sample_index,
                "layer": layer,
                "position": _metadata_value(metadata, sample_index, "token_position", "unknown"),
                "direction_type": direction_type,
                "latent_score": float(score),
                "label": int(dataset.labels[sample_index]),
                "passed": bool(dataset.labels[sample_index]),
                "score_auc_for_layer": auc,
            }
            if not metadata.empty:
                for column in metadata.columns:
                    if column not in row:
                        row[column] = metadata.iloc[sample_index][column]
            rows.append(row)
    return pd.DataFrame(rows)


def build_direction_control(
    dataset: ActivationDataset,
    *,
    direction_type: str,
    seed: int = 1234,
) -> dict[int, np.ndarray]:
    if direction_type == "correctness_direction":
        return compute_correctness_direction(dataset.activations, dataset.labels, layers=dataset.layers)
    if direction_type == "negative_correctness_direction":
        return {
            layer: -direction
            for layer, direction in compute_correctness_direction(dataset.activations, dataset.labels, layers=dataset.layers).items()
        }
    if direction_type == "random_direction":
        rng = np.random.default_rng(seed)
        return {
            layer: normalize(rng.normal(size=dataset.hidden_size).astype(np.float32))
            for layer in dataset.layers
        }
    if direction_type == "shuffled_label_direction":
        groups = dataset.metadata["task_id"].to_numpy(dtype=object) if "task_id" in dataset.metadata.columns else None
        shuffled = shuffle_labels_within_groups(dataset.labels, groups=groups, seed=seed).astype(np.int64)
        return compute_correctness_direction(dataset.activations, shuffled, layers=dataset.layers)
    if direction_type == "length_direction":
        return compute_length_direction(dataset)
    raise ValueError(f"direction_type desconhecido: {direction_type}")


def compute_length_direction(dataset: ActivationDataset) -> dict[int, np.ndarray]:
    if "generated_tokens" not in dataset.metadata.columns:
        return {}
    lengths = dataset.metadata["generated_tokens"].fillna(0).to_numpy(dtype=np.float32)
    if len(np.unique(lengths)) < 2:
        return {}
    high = lengths >= float(np.median(lengths))
    if high.all() or (~high).all():
        return {}
    directions = {}
    for layer_position, layer in enumerate(dataset.layers):
        x = dataset.activations[:, layer_position, :]
        directions[layer] = normalize(x[high].mean(axis=0) - x[~high].mean(axis=0))
    return directions


def summarize_latent_scores(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "layer",
                "direction_type",
                "mean_correct_score",
                "mean_incorrect_score",
                "score_gap",
                "auc",
                "n_samples",
            ]
        )
    rows = []
    for (layer, direction_type), group in frame.groupby(["layer", "direction_type"], sort=False):
        correct = group.loc[group["label"].eq(1), "latent_score"]
        incorrect = group.loc[group["label"].eq(0), "latent_score"]
        rows.append(
            {
                "layer": layer,
                "direction_type": direction_type,
                "mean_correct_score": float(correct.mean()) if len(correct) else np.nan,
                "mean_incorrect_score": float(incorrect.mean()) if len(incorrect) else np.nan,
                "score_gap": (
                    float(correct.mean() - incorrect.mean())
                    if len(correct) and len(incorrect)
                    else np.nan
                ),
                "auc": roc_auc_score(group["label"].to_numpy(dtype=int), group["latent_score"].to_numpy(dtype=float)),
                "n_samples": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0 or np.isnan(norm):
        return vector
    return (vector / norm).astype(np.float32)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float32)
    if x.ndim == 1:
        return normalize(x)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[(norms == 0) | np.isnan(norms)] = 1.0
    return (x / norms).astype(np.float32)


def _metadata_value(metadata: pd.DataFrame, row: int, column: str, default: Any) -> Any:
    if metadata.empty or column not in metadata.columns:
        return default
    return metadata.iloc[row][column]
