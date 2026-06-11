from __future__ import annotations

from slm_steering.policies.base import PolicyDecision, PolicyState, SamplingPolicy, StopReason, continue_decision, stop_decision


class DifficultyAdaptivePolicy(SamplingPolicy):
    name = "difficulty_adaptive"

    def __init__(self, easy_n: int = 1, fragile_n: int = 5, hard_n: int = 3, default_n: int = 5):
        self.budgets = {
            "easy": easy_n,
            "sampling_sensitive": default_n,
            "fragile": fragile_n,
            "hard": hard_n,
        }
        self.default_n = default_n

    def decide(self, state: PolicyState) -> PolicyDecision:
        if state.solved or state.last_passed:
            return stop_decision(StopReason.STOP_SUCCESS)
        budget = min(self.budgets.get(state.difficulty or "", self.default_n), state.max_attempts)
        if state.attempt_index >= budget:
            return stop_decision(StopReason.STOP_BUDGET_EXHAUSTED)
        return continue_decision()
