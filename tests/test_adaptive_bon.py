from __future__ import annotations

import unittest

from scripts.run_adaptive_bon import adaptive_metrics, build_policy
from slm_steering.policies import LatentScoreAdaptivePolicy, VerifierEarlyStopPolicy


class AdaptiveBestOfNTests(unittest.TestCase):
    def test_build_policy_known_names(self) -> None:
        self.assertIsInstance(build_policy("latent_adaptive", 3), LatentScoreAdaptivePolicy)
        self.assertIsInstance(build_policy("verifier_early_stop", 3), VerifierEarlyStopPolicy)

    def test_adaptive_metrics_reports_budget_saved(self) -> None:
        records = [
            {
                "solved": True,
                "attempts": [
                    {"generated_tokens": 10},
                    {"generated_tokens": 20},
                ],
            },
            {
                "solved": False,
                "attempts": [
                    {"generated_tokens": 30},
                ],
            },
        ]

        metrics = adaptive_metrics(records, fixed_n=3)

        self.assertEqual(metrics["success_rate"], 0.5)
        self.assertAlmostEqual(metrics["mean_attempts"], 1.5)
        self.assertAlmostEqual(metrics["budget_saved_vs_fixed_n"], 0.5)
        self.assertEqual(metrics["tokens_per_success"], 60)


if __name__ == "__main__":
    unittest.main()
