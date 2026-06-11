from __future__ import annotations

import unittest
import os

from slm_steering.activations.intervention import apply_additive_intervention, resolve_token_indices


class InterventionTests(unittest.TestCase):
    def test_token_index_resolution(self) -> None:
        self.assertEqual(resolve_token_indices("last", seq_len=5), [-1])
        self.assertEqual(resolve_token_indices("index:-2", seq_len=5), [3])
        self.assertEqual(resolve_token_indices("indices:0,-1", seq_len=5), [0, 4])

    def test_additive_intervention_updates_only_selected_tokens(self) -> None:
        torch = _require_torch()
        hidden = torch.zeros((1, 3, 4))
        direction = torch.tensor([1.0, 2.0, 3.0, 4.0])

        updated = apply_additive_intervention(hidden, direction, alpha=0.5, token_selection="index:1")

        self.assertTrue(torch.allclose(updated[0, 0], torch.zeros(4)))
        self.assertTrue(torch.allclose(updated[0, 1], direction * 0.5))
        self.assertTrue(torch.allclose(updated[0, 2], torch.zeros(4)))
        self.assertTrue(torch.allclose(hidden, torch.zeros_like(hidden)))

    def test_zero_alpha_returns_same_tensor(self) -> None:
        torch = _require_torch()
        hidden = torch.ones((1, 2, 3))
        updated = apply_additive_intervention(hidden, torch.ones(3), alpha=0.0)
        self.assertIs(updated, hidden)


def _require_torch():
    if os.environ.get("SLM_STEERING_TEST_TORCH") != "1":
        raise unittest.SkipTest("defina SLM_STEERING_TEST_TORCH=1 para testar operacoes com torch")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        raise unittest.SkipTest(f"torch indisponivel: {exc}") from exc
    return torch


if __name__ == "__main__":
    unittest.main()
