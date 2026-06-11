from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import numpy as np

from slm_steering.activations.controls import load_direction_map, normalize, random_direction_map
from slm_steering.activations.storage import save_activation_dataset


class ControlDirectionTests(unittest.TestCase):
    def test_random_direction_map_is_unit_normalized(self) -> None:
        directions = random_direction_map([1, 3], hidden_size=5, seed=7)

        self.assertEqual(set(directions), {1, 3})
        for vector in directions.values():
            self.assertEqual(vector.shape, (5,))
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_negative_correctness_direction_inverts_saved_direction(self) -> None:
        output_dir = Path("runs/tmp_tests/control_directions")
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(
                output_dir / "latent_directions.npz",
                layers=np.asarray([2], dtype=np.int32),
                directions=np.asarray([[3.0, 4.0]], dtype=np.float32),
            )
            positive = load_direction_map(
                output_dir,
                direction_type="correctness_direction",
                layers=[2],
                hidden_size=2,
            )
            negative = load_direction_map(
                output_dir,
                direction_type="negative_correctness_direction",
                layers=[2],
                hidden_size=2,
            )

            self.assertTrue(np.allclose(negative[2], -positive[2]))
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_length_control_uses_generated_token_metadata(self) -> None:
        output_dir = Path("runs/tmp_tests/length_direction")
        shutil.rmtree(output_dir, ignore_errors=True)
        activations = np.asarray(
            [
                [[0.0, 0.0]],
                [[0.1, 0.0]],
                [[2.0, 0.0]],
                [[2.2, 0.0]],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        metadata = [
            {"generated_tokens": 10, "passed": False},
            {"generated_tokens": 11, "passed": False},
            {"generated_tokens": 50, "passed": True},
            {"generated_tokens": 51, "passed": True},
        ]
        try:
            save_activation_dataset(
                output_dir,
                activations=activations,
                labels=labels,
                layers=[4],
                metadata=metadata,
                manifest={"phase": "test"},
            )
            directions = load_direction_map(
                output_dir,
                direction_type="length_direction",
                layers=[4],
                hidden_size=2,
            )

            self.assertTrue(np.allclose(directions[4], normalize(np.asarray([2.05, 0.0], dtype=np.float32))))
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
