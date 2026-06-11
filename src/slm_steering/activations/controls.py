from __future__ import annotations

from pathlib import Path

import numpy as np

from slm_steering.activations.probing import latent_directions
from slm_steering.activations.storage import load_activation_dataset


DIRECTION_TYPES = {
    "correctness_direction",
    "negative_correctness_direction",
    "random_direction",
    "shuffled_label_direction",
    "length_direction",
}


def load_direction_map(
    directions_dir: Path | None,
    *,
    direction_type: str,
    layers: list[int],
    hidden_size: int,
    seed: int = 1234,
    allow_missing: bool = True,
) -> dict[int, np.ndarray]:
    if direction_type not in DIRECTION_TYPES:
        raise ValueError(f"direction_type invalido: {direction_type}")
    if direction_type == "random_direction":
        return random_direction_map(layers, hidden_size=hidden_size, seed=seed)

    base = _load_base_directions(directions_dir, hidden_size=hidden_size)
    if direction_type == "shuffled_label_direction":
        base = _shuffled_label_directions(directions_dir, seed=seed) or base
    if direction_type == "length_direction":
        length = _length_directions(directions_dir)
        base = length or base
    if direction_type == "negative_correctness_direction":
        base = {layer: -vector for layer, vector in base.items()}

    result = {}
    for layer in layers:
        vector = base.get(layer)
        if vector is None:
            if not allow_missing:
                raise FileNotFoundError(f"Direcao ausente para camada {layer} em {directions_dir}")
            vector = np.zeros(hidden_size, dtype=np.float32)
        result[layer] = normalize(vector.astype(np.float32))
    return result


def random_direction_map(layers: list[int], *, hidden_size: int, seed: int = 1234) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {layer: normalize(rng.normal(size=hidden_size).astype(np.float32)) for layer in layers}


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0 or np.isnan(norm):
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _load_base_directions(directions_dir: Path | None, *, hidden_size: int) -> dict[int, np.ndarray]:
    if directions_dir is None:
        return {}
    path = directions_dir / "latent_directions.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    layers = [int(layer) for layer in data["layers"].tolist()]
    directions = data["directions"]
    if directions.ndim != 2 or directions.shape[1] != hidden_size:
        return {}
    return {layer: normalize(directions[index].astype(np.float32)) for index, layer in enumerate(layers)}


def _shuffled_label_directions(directions_dir: Path | None, *, seed: int) -> dict[int, np.ndarray]:
    if directions_dir is None or not (directions_dir / "activations.npz").exists():
        return {}
    dataset = load_activation_dataset(directions_dir)
    rng = np.random.default_rng(seed)
    shuffled = dataset.labels.copy()
    rng.shuffle(shuffled)
    fake = type(dataset)(
        activations=dataset.activations,
        labels=shuffled,
        layers=dataset.layers,
        metadata=dataset.metadata,
        manifest=dataset.manifest,
    )
    return {layer: normalize(vector) for layer, vector in latent_directions(fake).items()}


def _length_directions(directions_dir: Path | None) -> dict[int, np.ndarray]:
    if directions_dir is None or not (directions_dir / "activations.npz").exists():
        return {}
    dataset = load_activation_dataset(directions_dir)
    if "generated_tokens" not in dataset.metadata.columns:
        return {}
    lengths = dataset.metadata["generated_tokens"].fillna(0).to_numpy(dtype=np.float32)
    if len(np.unique(lengths)) < 2:
        return {}
    threshold = float(np.median(lengths))
    high = lengths >= threshold
    directions = {}
    for layer_position, layer in enumerate(dataset.layers):
        x = dataset.activations[:, layer_position, :]
        if high.all() or (~high).all():
            continue
        directions[layer] = normalize(x[high].mean(axis=0) - x[~high].mean(axis=0))
    return directions
