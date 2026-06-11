from __future__ import annotations

from slm_steering.policies.base import PolicyDecision, PolicyState, SamplingPolicy, StopReason, continue_decision, stop_decision


class VerifierEarlyStopPolicy(SamplingPolicy):
    name = "verifier_early_stop"

    def decide(self, state: PolicyState) -> PolicyDecision:
        if state.solved or state.last_passed:
            return stop_decision(StopReason.STOP_SUCCESS)
        if state.attempt_index >= state.max_attempts:
            return stop_decision(StopReason.STOP_BUDGET_EXHAUSTED)
        return continue_decision()
