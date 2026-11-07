from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def parse_layer_spec(spec: str, *, num_layers: int | None = None) -> list[int]:
    value = spec.strip().lower()
    if value == "all":
        if num_layers is None:
            raise ValueError("'all' exige num_layers")
        return list(range(num_layers))
    if ":" in value:
        parts = [int(part) if part else None for part in value.split(":")]
        if len(parts) > 3:
            raise ValueError(f"Especificacao de camadas invalida: {spec}")
        start = parts[0] if parts[0] is not None else 0
        stop = parts[1] if len(parts) > 1 and parts[1] is not None else num_layers
        step = parts[2] if len(parts) > 2 and parts[2] is not None else 1
        if stop is None:
            raise ValueError(f"Especificacao aberta exige num_layers: {spec}")
        if num_layers is not None:
            stop = min(stop, num_layers)
        return list(range(start, stop, step))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def resolve_transformer_layers(model: Any) -> Sequence[Any]:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for first, second in candidates:
        parent = getattr(model, first, None)
        layers = getattr(parent, second, None) if parent is not None else None
        if layers is not None:
            return layers
    direct_layers = getattr(model, "layers", None)
    if direct_layers is not None:
        return direct_layers
    raise ValueError("Nao foi possivel localizar as camadas transformer do modelo.")


class ActivationHookSet:
    def __init__(self, model: Any, layer_indices: list[int], token_index: int):
        self.model = model
        self.layer_indices = layer_indices
        self.token_index = token_index
        self.handles = []
        self.activations: dict[int, np.ndarray] = {}

    def __enter__(self) -> "ActivationHookSet":
        layers = resolve_transformer_layers(self.model)
        for layer_index in self.layer_indices:
            if layer_index < 0 or layer_index >= len(layers):
                raise IndexError(f"Camada fora do intervalo: {layer_index}")
            handle = layers[layer_index].register_forward_hook(self._make_hook(layer_index))
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            vector = hidden[0, self.token_index, :].detach().float().cpu().numpy()
            self.activations[layer_index] = vector

        return hook
