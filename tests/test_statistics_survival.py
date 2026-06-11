from __future__ import annotations

import unittest

from slm_steering.statistics.survival import (
    build_survival_table,
    expected_attempts_to_success,
    hazard_by_attempt,
    kaplan_meier_curve,
    tokens_until_first_success,
)


class SurvivalStatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "task_id": "a",
                "solved": True,
                "first_success_attempt": 2,
                "attempts": [
                    {"passed": False, "generated_tokens": 10},
                    {"passed": True, "generated_tokens": 20},
                ],
            },
            {
                "task_id": "b",
                "solved": False,
                "first_success_attempt": None,
                "attempts": [
                    {"passed": False, "generated_tokens": 5},
                    {"passed": False, "generated_tokens": 5},
                ],
            },
        ]

    def test_survival_table_marks_censoring(self) -> None:
        table = build_survival_table(self.records)

        self.assertEqual(int(table["event_observed"].sum()), 1)
        self.assertEqual(table.loc[table["task_id"].eq("b"), "time"].iloc[0], 2)

    def test_kaplan_meier_and_hazard(self) -> None:
        curve = kaplan_meier_curve(self.records)
        hazard = hazard_by_attempt(self.records)

        self.assertIn("cumulative_resolution", curve.columns)
        self.assertAlmostEqual(float(hazard.loc[hazard["attempt"].eq(2), "hazard"].iloc[0]), 0.5)

    def test_expected_attempts_censored_modes(self) -> None:
        budget = expected_attempts_to_success(self.records, censored="budget")
        ignore = expected_attempts_to_success(self.records, censored="ignore")
        tokens = tokens_until_first_success(self.records, censored="budget")

        self.assertEqual(budget["mean_attempts"], 2.0)
        self.assertEqual(ignore["mean_attempts"], 2.0)
        self.assertEqual(tokens["mean_tokens"], 20.0)


if __name__ == "__main__":
    unittest.main()
