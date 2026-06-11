from __future__ import annotations

from slm_steering.policies.base import PolicyDecision, PolicyState
from slm_steering.policies.latent_adaptive import LatentScoreAdaptivePolicy


class SteeringAdaptivePolicy(LatentScoreAdaptivePolicy):
    name = "steering_latent_adaptive"

    def decide(self, state: PolicyState) -> PolicyDecision:
        return super().decide(state)
