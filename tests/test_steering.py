from __future__ import annotations

import types
import unittest
import os

from slm_steering.activations.intervention import SteeringHookSet, SteeringInterventionConfig
from slm_steering.activations.layer_sweep import build_sweep


class SteeringTests(unittest.TestCase):
    def test_build_sweep_cartesian_product(self) -> None:
        sweep = build_sweep([1, 2], [0.0, 1.5], ["correctness_direction", "random_direction"])

        self.assertEqual(len(sweep), 8)
        self.assertEqual(sweep[1].label, "layer1_random_direction_alpha0p0")

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
