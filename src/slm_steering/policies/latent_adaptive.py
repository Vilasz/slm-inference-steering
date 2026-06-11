from __future__ import annotations

import math

from slm_steering.policies.base import PolicyDecision, PolicyState, SamplingPolicy, StopReason, continue_decision, stop_decision


class LatentScoreAdaptivePolicy(SamplingPolicy):
    name = "latent_adaptive"

    def __init__(
        self,
        *,
        value_success: float = 1000.0,
        token_cost_weight: float = 1.0,
        min_probability: float = 0.05,
    ):
        self.value_success = value_success
        self.token_cost_weight = token_cost_weight
        self.min_probability = min_probability

    def decide(self, state: PolicyState) -> PolicyDecision:
        if state.solved or state.last_passed:
            return stop_decision(StopReason.STOP_SUCCESS)
        if state.attempt_index >= state.max_attempts:
            return stop_decision(StopReason.STOP_BUDGET_EXHAUSTED)
        probability = estimate_success_probability(
            latent_score=state.latent_score,
            difficulty=state.difficulty,
            attempt_index=state.attempt_index,
        )
        expected_gain = probability * self.value_success
        marginal_cost = state.mean_token_cost * self.token_cost_weight
        if probability < self.min_probability or expected_gain <= marginal_cost:
            return stop_decision(StopReason.STOP_LOW_EXPECTED_GAIN, expected_gain=expected_gain)
        return continue_decision(expected_gain=expected_gain)


def estimate_success_probability(
    *,
    latent_score: float | None,
    difficulty: str | None,
    attempt_index: int,
) -> float:
    priors = {
        "easy": 0.65,
        "sampling_sensitive": 0.35,
        "fragile": 0.18,
        "hard": 0.05,
        None: 0.20,
    }
    base = priors.get(difficulty, 0.20)
    latent = 0.0 if latent_score is None else latent_score
    logit = math.log(base / max(1e-6, 1 - base)) + 0.8 * latent - 0.25 * max(0, attempt_index - 1)
    return 1 / (1 + math.exp(-logit))
