from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slm_steering.activations.probing import (
    best_layer_candidate,
    centroid_separability_frame,
    latent_directions,
    probe_layers,
)
from slm_steering.activations.storage import load_activation_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisa separabilidade de ativacoes da Fase 3.")
    parser.add_argument("activation_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--probe-steps", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_activation_dataset(args.activation_dir)
    output_dir = args.output_dir or args.activation_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    separability = centroid_separability_frame(dataset)
    probes = probe_layers(dataset, steps=args.probe_steps)
    directions = latent_directions(dataset)
    best_layer = best_layer_candidate(separability, probes)

    separability_path = output_dir / "separability.csv"
    probes_path = output_dir / "linear_probes.csv"
    directions_path = output_dir / "latent_directions.npz"
    report_path = output_dir / "activation_report.json"

    separability.to_csv(separability_path, index=False)
    probes.to_csv(probes_path, index=False)
    np.savez_compressed(
        directions_path,
        layers=np.asarray(list(directions.keys()), dtype=np.int32),
        directions=np.stack(list(directions.values())) if directions else np.zeros((0, 0)),
    )
    report = {
        "num_samples": dataset.num_samples,
        "layers": dataset.layers,
        "n_correct": int(dataset.labels.sum()),
        "n_incorrect": int((dataset.labels == 0).sum()),
        "best_layer_candidate": best_layer,
        "separability_csv": str(separability_path),
        "linear_probes_csv": str(probes_path),
        "latent_directions_npz": str(directions_path),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
