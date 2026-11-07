from __future__ import annotations

import unittest

from slm_steering.benchmark_registry import get_benchmark, list_benchmarks, load_benchmark
from slm_steering.benchmarks.stress import load_humaneval_stress


class StressBenchmarkTests(unittest.TestCase):
    def test_stress_loader_returns_coding_problems(self) -> None:
        problems = load_humaneval_stress(limit=2)

        self.assertEqual(len(problems), 2)
        self.assertTrue(problems[0].task_id.startswith("HumanEvalStress/"))
        self.assertIn("def ", problems[0].prompt)
        self.assertIn("def check", problems[0].test)

    def test_stress_benchmark_is_registered(self) -> None:
        spec = get_benchmark("humaneval_stress")
        problems = load_benchmark("humaneval_stress", limit=1)

        self.assertEqual(spec.status, "implemented")
        self.assertEqual(len(problems), 1)

    def test_core_benchmarks_are_registered_as_implemented(self) -> None:
        statuses = {spec.key: spec.status for spec in list_benchmarks()}

        for key in ["humaneval", "humaneval_plus", "mbpp", "humaneval_stress"]:
            self.assertEqual(statuses[key], "implemented")


if __name__ == "__main__":
    unittest.main()
