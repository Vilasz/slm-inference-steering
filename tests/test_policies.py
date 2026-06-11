from __future__ import annotations

import unittest

from slm_steering.policies import (
    DifficultyAdaptivePolicy,
    FixedNPolicy,
    LatentScoreAdaptivePolicy,
    PolicyState,
    StopReason,
    VerifierEarlyStopPolicy,
)
from slm_steering.policies.latent_adaptive import estimate_success_probability


class PolicyTests(unittest.TestCase):
    def test_fixed_n_policy_respects_budget(self) -> None:
        policy = FixedNPolicy(3)

        self.assertTrue(policy.decide(PolicyState(attempt_index=1, max_attempts=10)).continue_sampling)
        decision = policy.decide(PolicyState(attempt_index=3, max_attempts=10))

        self.assertFalse(decision.continue_sampling)
        self.assertEqual(decision.reason, StopReason.STOP_BUDGET_EXHAUSTED)

    def test_verifier_early_stop_policy_stops_on_success(self) -> None:
        policy = VerifierEarlyStopPolicy()
        decision = policy.decide(PolicyState(attempt_index=1, max_attempts=5, last_passed=True))

        self.assertFalse(decision.continue_sampling)
        self.assertEqual(decision.reason, StopReason.STOP_SUCCESS)

    def test_difficulty_adaptive_policy_allocates_more_budget_to_fragile_tasks(self) -> None:
        policy = DifficultyAdaptivePolicy(default_n=5)

        easy = policy.decide(PolicyState(attempt_index=2, max_attempts=5, difficulty="easy"))
        fragile = policy.decide(PolicyState(attempt_index=2, max_attempts=5, difficulty="fragile"))

        self.assertFalse(easy.continue_sampling)
        self.assertTrue(fragile.continue_sampling)

    def test_latent_probability_increases_with_positive_score(self) -> None:
        low = estimate_success_probability(latent_score=-2.0, difficulty="sampling_sensitive", attempt_index=1)
        high = estimate_success_probability(latent_score=2.0, difficulty="sampling_sensitive", attempt_index=1)

        self.assertGreater(high, low)

    def test_latent_policy_stops_when_expected_gain_is_low(self) -> None:
        policy = LatentScoreAdaptivePolicy(value_success=10.0, token_cost_weight=1.0, min_probability=0.2)
        decision = policy.decide(
            PolicyState(
                attempt_index=1,
                max_attempts=5,
                difficulty="hard",
                latent_score=-4.0,
                mean_token_cost=100.0,
            )
        )

        self.assertFalse(decision.continue_sampling)
        self.assertEqual(decision.reason, StopReason.STOP_LOW_EXPECTED_GAIN)


if __name__ == "__main__":
    unittest.main()
