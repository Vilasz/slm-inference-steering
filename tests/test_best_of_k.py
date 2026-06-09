from __future__ import annotations

import unittest

from slm_steering.best_of_k import best_of_k_marginal_curve, solved_by_sampling_tasks


def make_record(task_id: str, passed: list[bool]) -> dict:
    first_success = next((index + 1 for index, value in enumerate(passed) if value), None)
    return {
        "task_id": task_id,
        "entry_point": "f",
        "first_success_attempt": first_success,
        "attempts": [
            {
                "attempt": index + 1,
                "passed": value,
                "generated_tokens": 10,
                "generation_seconds": 1.0,
                "verification_seconds": 0.1,
            }
            for index, value in enumerate(passed)
        ],
    }


class BestOfKTests(unittest.TestCase):
    def test_best_of_k_curve_and_marginal_gain(self) -> None:
        records = [
            make_record("a", [False, True, False]),
            make_record("b", [False, False, False]),
            make_record("c", [True]),
        ]

        curve = best_of_k_marginal_curve(records, max_k=3)

        self.assertEqual(list(curve["k"]), [1, 2, 3])
        self.assertAlmostEqual(curve.loc[0, "best_of_k_accuracy"], 1 / 3)
        self.assertAlmostEqual(curve.loc[1, "best_of_k_accuracy"], 2 / 3)
        self.assertAlmostEqual(curve.loc[1, "marginal_gain_k"], 1 / 3)

    def test_stop_at_success_reduces_token_cost(self) -> None:
        records = [make_record("a", [False, True, False])]

        full = best_of_k_marginal_curve(records, max_k=3, stop_at_success=False)
        stopped = best_of_k_marginal_curve(records, max_k=3, stop_at_success=True)

        self.assertEqual(full.loc[2, "mean_tokens_until_k"], 30)
        self.assertEqual(stopped.loc[2, "mean_tokens_until_k"], 20)

    def test_solved_by_sampling_tasks(self) -> None:
        records = [
            make_record("sampling", [False, True]),
            make_record("pass1", [True]),
            make_record("hard", [False, False]),
        ]

        self.assertEqual(solved_by_sampling_tasks(records), ["sampling"])


if __name__ == "__main__":
    unittest.main()
