from __future__ import annotations

import unittest

import pandas as pd

from slm_steering.analysis import RunBundle
from slm_steering.pareto import pareto_efficient_mask, pareto_frame


class ParetoTests(unittest.TestCase):
    def test_pareto_mask_marks_dominated_points(self) -> None:
        frame = pd.DataFrame(
            [
                {"run": "cheap_good", "cost": 100, "quality": 0.8},
                {"run": "cheap_weaker", "cost": 90, "quality": 0.7},
                {"run": "expensive_same", "cost": 120, "quality": 0.8},
                {"run": "expensive_best", "cost": 130, "quality": 0.9},
            ]
        )

        mask = pareto_efficient_mask(frame, cost_col="cost", quality_col="quality")

        self.assertEqual(mask, [True, True, False, True])

    def test_pareto_frame_reads_run_metadata(self) -> None:
        run = RunBundle(
            label="run",
            records=[],
            summary={
                "num_tasks": 2,
                "observed_best_of_n": 0.5,
                "strict_pass_at_1": 0.0,
                "total_generated_tokens": 100,
                "total_generation_seconds": 5.0,
                "total_verification_seconds": 1.0,
                "mean_tokens_until_success_or_budget": 40,
                "mean_seconds_until_success_or_budget": 2.0,
                "mean_attempts_until_success_or_budget": 2,
                "total_attempts": 4,
                "solved_tasks": 1,
                "phase2": {
                    "model_id": "model",
                    "model_key": "model_key",
                    "benchmark": "humaneval",
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "n": 5,
                    "early_stop": False,
                },
            },
        )

        frame = pareto_frame([run])

        self.assertEqual(frame.loc[0, "model_key"], "model_key")
        self.assertEqual(frame.loc[0, "accuracy"], 0.5)
        self.assertEqual(frame.loc[0, "mean_generated_tokens"], 50)
        self.assertTrue(frame.loc[0, "is_pareto_efficient"])


if __name__ == "__main__":
    unittest.main()
