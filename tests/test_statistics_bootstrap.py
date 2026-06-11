from __future__ import annotations

import unittest

import pandas as pd

from slm_steering.statistics.bootstrap import (
    bootstrap_confidence_interval,
    bootstrap_metric_distribution,
    stratified_task_bootstrap,
)


class BootstrapStatisticsTests(unittest.TestCase):
    def test_stratified_bootstrap_preserves_task_groups(self) -> None:
        frame = pd.DataFrame(
            [
                {"task_id": "a", "attempt": 1, "passed": False},
                {"task_id": "a", "attempt": 2, "passed": True},
                {"task_id": "b", "attempt": 1, "passed": False},
                {"task_id": "b", "attempt": 2, "passed": False},
            ]
        )

        def metric(sample: pd.DataFrame) -> float:
            counts = sample.groupby("task_id").size()
            self.assertTrue(all(count % 2 == 0 for count in counts))
            return float(sample["passed"].mean())

        result = stratified_task_bootstrap(frame, metric, n_bootstrap=20, seed=7)

        self.assertEqual(result["n_bootstrap"], 20)
        self.assertLessEqual(result["ci_low"], result["ci_high"])

    def test_distribution_and_ci(self) -> None:
        frame = pd.DataFrame(
            [
                {"task_id": "a", "value": 0.0},
                {"task_id": "b", "value": 1.0},
                {"task_id": "c", "value": 1.0},
            ]
        )
        distribution = bootstrap_metric_distribution(
            frame,
            lambda sample: float(sample["value"].mean()),
            n_bootstrap=30,
            seed=3,
        )
        ci = bootstrap_confidence_interval(distribution)

        self.assertEqual(len(distribution), 30)
        self.assertEqual(ci.n_bootstrap, 30)


if __name__ == "__main__":
    unittest.main()
