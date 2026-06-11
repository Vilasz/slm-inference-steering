from __future__ import annotations

import unittest

import pandas as pd

from slm_steering.statistics.regression import build_success_regression_frame, compare_success_models


class RegressionStatisticsTests(unittest.TestCase):
    def test_compare_success_models_runs_on_synthetic_data(self) -> None:
        frame = pd.DataFrame(
            [
                {"task_id": "a", "passed": 0, "latent_score": -2.0, "generated_tokens": 100, "attempt": 1},
                {"task_id": "b", "passed": 0, "latent_score": -1.5, "generated_tokens": 120, "attempt": 1},
                {"task_id": "c", "passed": 0, "latent_score": -1.0, "generated_tokens": 110, "attempt": 2},
                {"task_id": "d", "passed": 0, "latent_score": -0.5, "generated_tokens": 130, "attempt": 2},
                {"task_id": "e", "passed": 1, "latent_score": 0.5, "generated_tokens": 90, "attempt": 1},
                {"task_id": "f", "passed": 1, "latent_score": 1.0, "generated_tokens": 95, "attempt": 1},
                {"task_id": "g", "passed": 1, "latent_score": 1.5, "generated_tokens": 80, "attempt": 2},
                {"task_id": "h", "passed": 1, "latent_score": 2.0, "generated_tokens": 85, "attempt": 2},
            ]
        )
        regression_frame = build_success_regression_frame(frame)
        results = compare_success_models(regression_frame, seed=4)

        self.assertIn("observables_plus_latent", set(results["model_name"]))
        self.assertIn("auc", results.columns)
        self.assertTrue(results["n_test_samples"].max() > 0)


if __name__ == "__main__":
    unittest.main()
