"""Exact small finite-budget MDP used only for conceptual/synthetic validation."""
from __future__ import annotations

from functools import lru_cache


def finite_budget_policy(success_probability, completion_cost, start_cost, penalty, max_attempts, budget):
    """Costs and budget are integer units; verification is included in completion_cost.

    At a checkpoint k is the number of *new* attempts left after the active one.
    Restart consumes start_cost and a slot; failure reveals no information about
    future attempts in this deliberately iid synthetic model.
    """
    if not 0 <= success_probability <= 1 or min(completion_cost, start_cost) <= 0 or penalty < 0:
        raise ValueError('Invalid synthetic decision model')
    if any(int(v) != v or v < 0 for v in (completion_cost, start_cost, max_attempts, budget)):
        raise ValueError('Finite solver uses nonnegative integer resource units')
    p = success_probability
    actions = {}

    @lru_cache(None)
    def idle(k, b):
        choices = {'stop': float(penalty)}
        if k > 0 and b >= start_cost:
            choices['start'] = start_cost + active(k - 1, b - start_cost)
        action = min(choices, key=lambda a: (choices[a], a))
        actions[('idle', k, b)] = {'action': action, 'value': choices[action], 'choices': choices}
        return choices[action]

    @lru_cache(None)
    def active(k, b):
        choices = {'abort': idle(k, b), 'stop': float(penalty)}
        if b >= completion_cost:
            choices['continue'] = completion_cost + (1 - p) * idle(k, b - completion_cost)
        action = min(choices, key=lambda a: (choices[a], a))
        actions[('active', k, b)] = {'action': action, 'value': choices[action], 'choices': choices}
        return choices[action]

    value = idle(max_attempts, budget)
    return {'value': value, 'states': [{'state': list(k), **v} for k, v in actions.items()]}


def simplified_continue(p, fallback_value, remaining_cost):
    """Compare c+(1-p)V with V, in one common cost unit, with equal fallback state."""
    if not 0 <= p <= 1 or min(fallback_value, remaining_cost) < 0:
        raise ValueError('Invalid decision quantities')
    return p * fallback_value >= remaining_cost
