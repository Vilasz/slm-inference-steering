from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slm_steering.activations.extraction import resolve_token_position
from slm_steering.activations.hooks import ActivationHookSet


@dataclass(frozen=True)
class LatentScoreConfig:
    directions_dir: Path
    layer: int | None = None
    token_position: str = "completion_last"


@dataclass(frozen=True)
class LatentDirectionScorer:
    layer: int
    direction: np.ndarray
    token_position: str = "completion_last"

    def score(self, generator, humaneval_prompt: str, completion: str) -> float:
        rendered_prompt = generator._render_prompt(humaneval_prompt)
        full_text = rendered_prompt + completion
        prompt_inputs = generator.tokenizer(rendered_prompt, return_tensors="pt")
        full_inputs = generator.tokenizer(full_text, return_tensors="pt").to(generator.device)
        prompt_tokens = int(prompt_inputs["input_ids"].shape[-1])
        total_tokens = int(full_inputs["input_ids"].shape[-1])
        token_index = resolve_token_position(
            self.token_position,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )
        with generator.torch.inference_mode():
            with ActivationHookSet(generator.model, [self.layer], token_index) as hook_set:
                generator.model(**full_inputs)
        activation = hook_set.activations[self.layer].astype(np.float32)
        return cosine_score(activation, self.direction)


def build_latent_direction_scorer(config: LatentScoreConfig) -> LatentDirectionScorer | None:
    path = config.directions_dir / "latent_directions.npz"
    if not path.exists():
        return None
    arrays = np.load(path)
    layers = [int(layer) for layer in arrays["layers"].tolist()]
    directions = arrays["directions"].astype(np.float32)
    if not layers or directions.ndim != 2:
        return None
    layer = config.layer if config.layer is not None else select_default_layer(config.directions_dir, layers)
    if layer not in layers:
        return None
    index = layers.index(layer)
    return LatentDirectionScorer(
        layer=layer,
        direction=normalize(directions[index]),
        token_position=config.token_position,
    )


def select_default_layer(directions_dir: Path, available_layers: list[int]) -> int:
    for filename, column in [
        ("linear_probes.csv", "test_accuracy"),
        ("separability.csv", "fisher_ratio"),
        ("separability.csv", "centroid_distance"),
    ]:
        path = directions_dir / filename
        if not path.exists():
            continue
        try:
            import pandas as pd

            frame = pd.read_csv(path)
        except Exception:
            continue
        if "layer" not in frame.columns or column not in frame.columns:
            continue
        frame = frame[frame["layer"].isin(available_layers)].copy()
        frame[column] = frame[column].fillna(float("-inf"))
        if not frame.empty:
            return int(frame.sort_values(column, ascending=False).iloc[0]["layer"])
    return available_layers[len(available_layers) // 2]


def cosine_score(activation: np.ndarray, direction: np.ndarray) -> float:
    left = normalize(activation.astype(np.float32))
    right = normalize(direction.astype(np.float32))
    if not left.any() or not right.any():
        return 0.0
    return float(np.dot(left, right))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0 or np.isnan(norm):
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)
