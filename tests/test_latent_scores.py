from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from slm_steering.activations.latent_scores import (
    build_direction_control,
    compute_correctness_direction,
    compute_layerwise_latent_scores,
    project_onto_direction,
)
from slm_steering.activations.storage import ActivationDataset


class LatentScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        activations = np.asarray(
            [
                [[0.0, 0.0], [0.1, 0.0]],
                [[0.2, 0.0], [0.0, 0.1]],
                [[2.0, 0.0], [1.0, 1.0]],
                [[2.2, 0.0], [1.1, 1.0]],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        metadata = pd.DataFrame(
            [
                {"task_id": "a", "attempt": 1, "generated_tokens": 10, "token_position": "completion_last"},
                {"task_id": "b", "attempt": 1, "generated_tokens": 11, "token_position": "completion_last"},
                {"task_id": "c", "attempt": 1, "generated_tokens": 50, "token_position": "completion_last"},
                {"task_id": "d", "attempt": 1, "generated_tokens": 51, "token_position": "completion_last"},
            ]
        )
        self.dataset = ActivationDataset(
            activations=activations,
            labels=labels,
            layers=[0, 1],
            metadata=metadata,
            manifest={"model_id": "synthetic"},
        )

    def test_correctness_direction_dimensions(self) -> None:
        directions = compute_correctness_direction(
            self.dataset.activations,
            self.dataset.labels,
            layers=self.dataset.layers,
        )

        self.assertEqual(set(directions), {0, 1})
        self.assertEqual(directions[0].shape, (2,))

    def test_projection_score_orders_examples(self) -> None:
        directions = compute_correctness_direction(
            self.dataset.activations,
            self.dataset.labels,
            layers=self.dataset.layers,
        )
        scores = project_onto_direction(self.dataset.activations[:, 0, :], directions[0])

        self.assertGreater(scores[2], scores[0])

    def test_layerwise_scores_and_controls(self) -> None:
        scores = compute_layerwise_latent_scores(self.dataset, direction_type="correctness_direction")
        random_directions = build_direction_control(self.dataset, direction_type="random_direction", seed=1)
        negative = build_direction_control(self.dataset, direction_type="negative_correctness_direction")

        self.assertFalse(scores.empty)
        self.assertEqual(set(scores["layer"]), {0, 1})
        self.assertEqual(set(random_directions), {0, 1})
        self.assertTrue(np.allclose(negative[0], -build_direction_control(self.dataset, direction_type="correctness_direction")[0]))


if __name__ == "__main__":
    unittest.main()
