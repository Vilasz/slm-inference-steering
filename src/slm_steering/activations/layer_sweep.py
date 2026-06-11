from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class SteeringSweepPoint:
    layer: int
    alpha: float
    direction_type: str

    @property
    def label(self) -> str:
        alpha = str(self.alpha).replace("-", "m").replace(".", "p")
        return f"layer{self.layer}_{self.direction_type}_alpha{alpha}"


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_sweep(layers: list[int], alphas: list[float], direction_types: list[str]) -> list[SteeringSweepPoint]:
    return [
        SteeringSweepPoint(layer=layer, alpha=alpha, direction_type=direction_type)
        for layer, alpha, direction_type in product(layers, alphas, direction_types)
    ]
