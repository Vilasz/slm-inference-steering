from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SteeringInterventionConfig:
    layer: int
    direction: Any
    alpha: float
    token_selection: str = "last"
    normalize_direction: bool = True
    clip_norm: float | None = None
    prompt_length: int | None = None


@dataclass(frozen=True)
class SteeringSchedule:
    layers: list[int]
    alphas: list[float]
    token_policy: str = "last"
    normalize: str = "unit_l2"
    clip_norm: float | None = None

    def interventions(self, direction_by_layer: dict[int, Any], *, prompt_length: int | None = None) -> list[SteeringInterventionConfig]:
        if len(self.alphas) == 1:
            alphas = self.alphas * len(self.layers)
        elif len(self.alphas) == len(self.layers):
            alphas = self.alphas
        else:
            raise ValueError("alphas deve ter tamanho 1 ou o mesmo tamanho de layers")
        return [
            SteeringInterventionConfig(
                layer=layer,
                direction=direction_by_layer[layer],
                alpha=alpha,
                token_selection=self.token_policy,
                normalize_direction=self.normalize == "unit_l2",
                clip_norm=self.clip_norm,
                prompt_length=prompt_length,
            )
            for layer, alpha in zip(self.layers, alphas)
        ]


def apply_additive_intervention(
    hidden,
    direction,
    *,
    alpha: float,
    token_selection: str = "last",
    normalize_direction: bool = True,
    clip_norm: float | None = None,
    prompt_length: int | None = None,
):
    if alpha == 0:
        return hidden
    import torch

    vector = torch.as_tensor(direction, dtype=hidden.dtype, device=hidden.device)
    if vector.ndim != 1:
        raise ValueError("direction deve ser um vetor 1D")
    if vector.shape[0] != hidden.shape[-1]:
        raise ValueError(f"direction hidden_size={vector.shape[0]} != hidden hidden_size={hidden.shape[-1]}")

    if normalize_direction:
        norm = torch.linalg.vector_norm(vector)
        if bool(norm > 0):
            vector = vector / norm
    if clip_norm is not None:
        norm = torch.linalg.vector_norm(vector)
        if bool(norm > clip_norm):
            vector = vector * (clip_norm / norm)

    update = alpha * vector
    result = hidden.clone()
    positions = resolve_token_indices(token_selection, seq_len=hidden.shape[1], prompt_length=prompt_length)
    result[:, positions, :] = result[:, positions, :] + update
    return result


def resolve_token_indices(token_selection: str, *, seq_len: int, prompt_length: int | None = None):
    value = token_selection.strip().lower()
    if value in {"all", "all_tokens"}:
        return slice(None)
    if value in {"last", "completion_last"}:
        return [-1]
    if value in {"generated", "generated_only"}:
        if prompt_length is None:
            return [-1]
        return list(range(min(prompt_length, seq_len), seq_len)) or [-1]
    if value in {"prompt", "prompt_tokens"}:
        if prompt_length is None:
            return list(range(max(0, seq_len - 1)))
        return list(range(min(prompt_length, seq_len)))
    if value == "prompt_last":
        if prompt_length is None:
            return [-1]
        return [max(0, min(prompt_length, seq_len) - 1)]
    if value == "first_generated":
        if prompt_length is None:
            return [-1]
        return [min(prompt_length, seq_len - 1)]
    if value.startswith("index:"):
        index = int(value.split(":", 1)[1])
        if index < 0:
            index = seq_len + index
        if index < 0 or index >= seq_len:
            raise IndexError(f"Indice fora da sequencia: {token_selection}")
        return [index]
    if value.startswith("indices:"):
        indices = [int(item) for item in value.split(":", 1)[1].split(",") if item]
        resolved = []
        for index in indices:
            if index < 0:
                index = seq_len + index
            if index < 0 or index >= seq_len:
                raise IndexError(f"Indice fora da sequencia: {index}")
            resolved.append(index)
        return resolved
    raise ValueError(f"token_selection invalido: {token_selection}")


class SteeringHookSet:
    def __init__(self, model: Any, interventions: list[SteeringInterventionConfig]):
        from slm_steering.activations.hooks import resolve_transformer_layers

        self.layers = resolve_transformer_layers(model)
        self.interventions = interventions
        self.handles = []

    def __enter__(self) -> "SteeringHookSet":
        grouped: dict[int, list[SteeringInterventionConfig]] = {}
        for intervention in self.interventions:
            grouped.setdefault(intervention.layer, []).append(intervention)
        for layer, interventions in grouped.items():
            if layer < 0 or layer >= len(self.layers):
                raise IndexError(f"Camada fora do intervalo: {layer}")
            handle = self.layers[layer].register_forward_hook(self._make_hook(interventions))
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, interventions: list[SteeringInterventionConfig]):
        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            for intervention in interventions:
                hidden = apply_additive_intervention(
                    hidden,
                    intervention.direction,
                    alpha=intervention.alpha,
                    token_selection=intervention.token_selection,
                    normalize_direction=intervention.normalize_direction,
                    clip_norm=intervention.clip_norm,
                    prompt_length=intervention.prompt_length,
                )
            if is_tuple:
                return (hidden, *output[1:])
            return hidden

        return hook
