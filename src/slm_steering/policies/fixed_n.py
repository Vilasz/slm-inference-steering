from __future__ import annotations

from slm_steering.policies.base import PolicyDecision, PolicyState, SamplingPolicy, StopReason, continue_decision, stop_decision


class FixedNPolicy(SamplingPolicy):
    name = "fixed_n"

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n deve ser >= 1")
        self.n = n

    def decide(self, state: PolicyState) -> PolicyDecision:
        if state.attempt_index >= min(self.n, state.max_attempts):
            return stop_decision(StopReason.STOP_BUDGET_EXHAUSTED)
        return continue_decision()
