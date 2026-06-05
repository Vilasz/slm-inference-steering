from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    family: str
    parameters_b: float
    use_chat_template: bool = True
    gpu_tier: str = "core_cuda_6gb"
    notes: str = ""


@dataclass(frozen=True)
class DecodingSpec:
    key: str
    n: int
    temperature: float
    top_p: float
    early_stop: bool = False
    notes: str = ""


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen2.5-coder-0.5b-instruct": ModelSpec(
        key="qwen2.5-coder-0.5b-instruct",
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        family="Qwen2.5-Coder",
        parameters_b=0.5,
        notes="Smaller sibling for scale and latency comparisons.",
    ),
    "qwen2.5-coder-1.5b-instruct": ModelSpec(
        key="qwen2.5-coder-1.5b-instruct",
        model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        family="Qwen2.5-Coder",
        parameters_b=1.54,
        notes="Primary TCC baseline already used in phase 1.",
    ),
    "deepseek-coder-1.3b-instruct": ModelSpec(
        key="deepseek-coder-1.3b-instruct",
        model_id="deepseek-ai/deepseek-coder-1.3b-instruct",
        family="DeepSeek-Coder",
        parameters_b=1.3,
        notes="External code-specialized baseline at similar scale.",
    ),
    "qwen2.5-coder-3b-instruct": ModelSpec(
        key="qwen2.5-coder-3b-instruct",
        model_id="Qwen/Qwen2.5-Coder-3B-Instruct",
        family="Qwen2.5-Coder",
        parameters_b=3.0,
        gpu_tier="stretch_cuda_6gb",
        notes="Optional stretch run; may require lower max_new_tokens on 6 GB VRAM.",
    ),
}


DECODING_SPECS: dict[str, DecodingSpec] = {
    "greedy_n1": DecodingSpec(
        key="greedy_n1",
        n=1,
        temperature=0.0,
        top_p=1.0,
        notes="Deterministic pass@1 baseline.",
    ),
    "sample_n5": DecodingSpec(
        key="sample_n5",
        n=5,
        temperature=0.8,
        top_p=0.95,
        notes="Best-of-5 sampling budget.",
    ),
    "sample_n5_early_stop": DecodingSpec(
        key="sample_n5_early_stop",
        n=5,
        temperature=0.8,
        top_p=0.95,
        early_stop=True,
        notes="Same budget with verifier-based stopping.",
    ),
    "conservative_n5": DecodingSpec(
        key="conservative_n5",
        n=5,
        temperature=0.2,
        top_p=0.9,
        notes="Lower-diversity sampling to test diversity vs correctness.",
    ),
}


PRESETS: dict[str, dict[str, list[str]]] = {
    "smoke": {
        "models": ["qwen2.5-coder-0.5b-instruct"],
        "decodings": ["greedy_n1", "sample_n5"],
    },
    "core": {
        "models": [
            "qwen2.5-coder-0.5b-instruct",
            "qwen2.5-coder-1.5b-instruct",
            "deepseek-coder-1.3b-instruct",
        ],
        "decodings": ["greedy_n1", "sample_n5", "sample_n5_early_stop"],
    },
    "full": {
        "models": [
            "qwen2.5-coder-0.5b-instruct",
            "qwen2.5-coder-1.5b-instruct",
            "deepseek-coder-1.3b-instruct",
            "qwen2.5-coder-3b-instruct",
        ],
        "decodings": ["greedy_n1", "sample_n5", "sample_n5_early_stop", "conservative_n5"],
    },
}


def model_specs(keys: Iterable[str]) -> list[ModelSpec]:
    return [_lookup(MODEL_SPECS, key, "model") for key in keys]


def decoding_specs(keys: Iterable[str]) -> list[DecodingSpec]:
    return [_lookup(DECODING_SPECS, key, "decoding") for key in keys]


def preset_keys(name: str) -> tuple[list[str], list[str]]:
    if name not in PRESETS:
        raise ValueError(f"Preset invalido: {name}. Opcoes: {', '.join(sorted(PRESETS))}")
    preset = PRESETS[name]
    return list(preset["models"]), list(preset["decodings"])


def parse_key_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    keys = [item.strip() for item in value.split(",") if item.strip()]
    return keys or None


def run_label(model: ModelSpec, decoding: DecodingSpec) -> str:
    return safe_name(f"{model.key}__{decoding.key}")


def safe_name(value: str) -> str:
    value = value.lower().replace("/", "_")
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-")


def _lookup(mapping: dict[str, object], key: str, kind: str):
    try:
        return mapping[key]
    except KeyError as exc:
        available = ", ".join(sorted(mapping))
        raise ValueError(f"{kind} desconhecido: {key}. Opcoes: {available}") from exc
