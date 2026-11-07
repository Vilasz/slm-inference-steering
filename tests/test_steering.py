from __future__ import annotations

import types
import unittest
import os

from slm_steering.activations.intervention import (
    SteeringHookSet,
    SteeringInterventionConfig,
    SteeringSchedule,
    apply_additive_intervention,
    resolve_token_indices,
)
from slm_steering.activations.layer_sweep import build_sweep


class SteeringTests(unittest.TestCase):
    def test_build_sweep_cartesian_product(self) -> None:
        sweep = build_sweep([1, 2], [0.0, 1.5], ["correctness_direction", "random_direction"])

        self.assertEqual(len(sweep), 8)
        self.assertEqual(sweep[1].label, "layer1_random_direction_alpha0p0")

    def test_token_policy_generated_only_uses_prompt_length(self) -> None:
        self.assertEqual(resolve_token_indices("generated_only", seq_len=5, prompt_length=3), [3, 4])
        self.assertEqual(resolve_token_indices("prompt_tokens", seq_len=5, prompt_length=3), [0, 1, 2])

    def test_steering_schedule_expands_single_alpha(self) -> None:
        schedule = SteeringSchedule(layers=[8, 12], alphas=[0.5], token_policy="generated_only", clip_norm=1.0)

        interventions = schedule.interventions({8: [1.0, 0.0], 12: [0.0, 1.0]}, prompt_length=4)

        self.assertEqual([item.layer for item in interventions], [8, 12])
        self.assertEqual([item.alpha for item in interventions], [0.5, 0.5])
        self.assertEqual(interventions[0].token_selection, "generated_only")
        self.assertEqual(interventions[0].prompt_length, 4)

    def test_hook_set_applies_intervention_to_dummy_layer(self) -> None:
        torch = _require_torch()

        class IdentityLayer(torch.nn.Module):
            def forward(self, x):
                return x

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = types.SimpleNamespace()
                self.model.layers = torch.nn.ModuleList([IdentityLayer()])

            def forward(self, x):
                return self.model.layers[0](x)

        model = DummyModel()
        hidden = torch.zeros((1, 2, 3))
        intervention = SteeringInterventionConfig(
            layer=0,
            direction=torch.tensor([0.0, 1.0, 0.0]),
            alpha=2.0,
            token_selection="last",
        )
        with SteeringHookSet(model, [intervention]):
            updated = model(hidden)

        self.assertTrue(torch.allclose(updated[0, 0], torch.zeros(3)))
        self.assertTrue(torch.allclose(updated[0, 1], torch.tensor([0.0, 2.0, 0.0])))

    def test_intervention_normalizes_and_clips_direction(self) -> None:
        torch = _require_torch()
        hidden = torch.zeros((1, 3, 2))

        updated = apply_additive_intervention(
            hidden,
            torch.tensor([3.0, 4.0]),
            alpha=2.0,
            token_selection="generated_only",
            normalize_direction=True,
            clip_norm=0.5,
            prompt_length=2,
        )

        self.assertTrue(torch.allclose(updated[0, 0], torch.zeros(2)))
        self.assertTrue(torch.allclose(updated[0, 1], torch.zeros(2)))
        self.assertTrue(torch.allclose(updated[0, 2], torch.tensor([0.6, 0.8]), atol=1e-6))


def _require_torch():
    if os.environ.get("SLM_STEERING_TEST_TORCH") != "1":
        raise unittest.SkipTest("defina SLM_STEERING_TEST_TORCH=1 para testar hooks com torch")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        raise unittest.SkipTest(f"torch indisponivel: {exc}") from exc
    return torch


if __name__ == "__main__":
    unittest.main()
