from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ActivationDataset:
    activations: np.ndarray
    labels: np.ndarray
    layers: list[int]
    metadata: pd.DataFrame
    manifest: dict[str, Any]

    @property
    def num_samples(self) -> int:
        return int(self.activations.shape[0])

    @property
    def num_layers(self) -> int:
        return int(self.activations.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.activations.shape[2])


def save_activation_dataset(
    output_dir: Path,
    *,
    activations: np.ndarray,
    labels: np.ndarray,
    layers: list[int],
    metadata: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "activations.npz"
    metadata_path = output_dir / "metadata.jsonl"
    manifest_path = output_dir / "manifest.json"

    np.savez_compressed(
        arrays_path,
        activations=activations,
        labels=labels.astype(np.int8),
        layers=np.asarray(layers, dtype=np.int32),
    )
    with metadata_path.open("w", encoding="utf-8") as file:
        for row in metadata:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        **manifest,
        "num_samples": int(activations.shape[0]),
        "num_layers": int(activations.shape[1]),
        "hidden_size": int(activations.shape[2]) if activations.size else None,
        "arrays_path": str(arrays_path),
        "metadata_path": str(metadata_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "arrays_path": str(arrays_path),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
    }


def load_activation_dataset(path: Path) -> ActivationDataset:
    root = path if path.is_dir() else path.parent
    manifest_path = root / "manifest.json"
    arrays_path = root / "activations.npz"
    metadata_path = root / "metadata.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = np.load(arrays_path)
    metadata_rows = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                metadata_rows.append(json.loads(line))
    return ActivationDataset(
        activations=arrays["activations"].astype(np.float32),
        labels=arrays["labels"].astype(np.int64),
        layers=[int(value) for value in arrays["layers"].tolist()],
        metadata=pd.DataFrame(metadata_rows),
        manifest=manifest,
    )
