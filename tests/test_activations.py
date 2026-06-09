from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import numpy as np

from slm_steering.activations.extraction import resolve_token_position
from slm_steering.activations.hooks import parse_layer_spec
from slm_steering.activations.probing import (
    best_layer_candidate,
    centroid_separability_frame,
    latent_directions,
    pca_2d,
    probe_layers,
)
from slm_steering.activations.storage import load_activation_dataset, save_activation_dataset


class ActivationHookUtilityTests(unittest.TestCase):
    def test_parse_layer_spec_list_and_range(self) -> None:
        self.assertEqual(parse_layer_spec("0,2,4"), [0, 2, 4])
        self.assertEqual(parse_layer_spec("0:10:3", num_layers=8), [0, 3, 6])
        self.assertEqual(parse_layer_spec("all", num_layers=3), [0, 1, 2])

    def test_resolve_token_position(self) -> None:
        self.assertEqual(resolve_token_position("completion_last", prompt_tokens=5, total_tokens=8), 7)
        self.assertEqual(resolve_token_position("prompt_last", prompt_tokens=5, total_tokens=8), 4)
        self.assertEqual(resolve_token_position("first_completion", prompt_tokens=5, total_tokens=8), 5)
        self.assertEqual(resolve_token_position("index:-1", prompt_tokens=5, total_tokens=8), 7)


class ActivationStorageAndProbeTests(unittest.TestCase):
    def test_save_load_and_probe_dataset(self) -> None:
        output_dir = Path("runs/tmp_tests/activations")
        shutil.rmtree(output_dir, ignore_errors=True)
        activations = np.asarray(
            [
                [[0.0, 0.0], [0.1, 0.1]],
                [[0.2, 0.0], [0.2, 0.1]],
                [[3.0, 3.0], [4.0, 4.0]],
                [[3.2, 3.1], [4.1, 4.2]],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        metadata = [
            {"task_id": f"task/{idx}", "attempt": 1, "passed": bool(label)}
            for idx, label in enumerate(labels)
        ]
        try:
            save_activation_dataset(
                output_dir,
                activations=activations,
                labels=labels,
                layers=[0, 1],
                metadata=metadata,
                manifest={"model_id": "synthetic"},
            )
            dataset = load_activation_dataset(output_dir)
            separability = centroid_separability_frame(dataset)
            probes = probe_layers(dataset, steps=50)
            directions = latent_directions(dataset)
            pca = pca_2d(dataset.activations[:, 0, :])

            self.assertEqual(dataset.num_samples, 4)
            self.assertEqual(dataset.layers, [0, 1])
            self.assertEqual(pca.shape, (4, 2))
            self.assertEqual(set(directions), {0, 1})
            self.assertGreater(separability["centroid_distance"].max(), 0)
            self.assertIn(best_layer_candidate(separability, probes), {0, 1})
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
