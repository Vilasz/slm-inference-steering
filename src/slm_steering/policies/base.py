from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StopReason(str, Enum):
    CONTINUE_SAMPLING = "continue_sampling"
    STOP_SUCCESS = "stop_success"
    STOP_LOW_EXPECTED_GAIN = "stop_low_expected_gain"
    STOP_BUDGET_EXHAUSTED = "stop_budget_exhausted"


@dataclass(frozen=True)
class PolicyState:
    attempt_index: int
    max_attempts: int
    solved: bool = False
    last_passed: bool = False
    difficulty: str | None = None
    latent_score: float | None = None
    generated_tokens: int = 0
    mean_token_cost: float = 128.0


@dataclass(frozen=True)
class PolicyDecision:
    continue_sampling: bool
    reason: StopReason
    expected_gain: float | None = None


class SamplingPolicy:
    name = "base"

    def decide(self, state: PolicyState) -> PolicyDecision:
        raise NotImplementedError


def continue_decision(expected_gain: float | None = None) -> PolicyDecision:
    return PolicyDecision(True, StopReason.CONTINUE_SAMPLING, expected_gain=expected_gain)


def stop_decision(reason: StopReason, expected_gain: float | None = None) -> PolicyDecision:
    return PolicyDecision(False, reason, expected_gain=expected_gain)
