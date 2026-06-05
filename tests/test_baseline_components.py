from __future__ import annotations

import unittest
from unittest.mock import patch

from slm_steering.analysis import (
    RunBundle,
    attempts_frame,
    bootstrap_success_table,
    counterfactual_early_stop_frame,
    diversity_frame,
    tasks_frame,
)
from slm_steering.datasets import DEFAULT_HUMANEVAL_DATASET, CodingProblem, load_humaneval
from slm_steering.metrics import estimate_pass_at_k, summarize
from slm_steering.reporting import write_run_artifacts
from slm_steering.verifier import build_candidate_source, check_correctness, normalize_completion
from scripts.run_baseline import run_problem


class VerifierTests(unittest.TestCase):
    def test_verifier_accepts_valid_completion(self) -> None:
        problem = CodingProblem(
            task_id="sample/0",
            prompt="def add_one(x):\n",
            test="def check(candidate):\n    assert candidate(1) == 2",
            entry_point="add_one",
        )

        result = check_correctness(problem, "    return x + 1", timeout_seconds=2)

        self.assertTrue(result.passed, result.stderr)

    def test_verifier_rejects_invalid_completion(self) -> None:
        problem = CodingProblem(
            task_id="sample/1",
            prompt="def add_one(x):\n",
            test="def check(candidate):\n    assert candidate(1) == 2",
            entry_point="add_one",
        )

        result = check_correctness(problem, "    return x", timeout_seconds=2)

        self.assertFalse(result.passed)

    def test_normalize_markdown_code_fence(self) -> None:
        completion = "Claro:\n```python\n    return x + 1\n```\n"

        self.assertEqual(normalize_completion(completion), "    return x + 1")

    def test_full_function_is_not_duplicated(self) -> None:
        problem = CodingProblem(
            task_id="sample/2",
            prompt="def add_one(x):\n",
            test="def check(candidate):\n    assert candidate(1) == 2",
            entry_point="add_one",
        )

        source = build_candidate_source(problem, "def add_one(x):\n    return x + 1")

        self.assertEqual(source.count("def add_one"), 1)

    def test_full_function_keeps_prompt_imports(self) -> None:
        problem = CodingProblem(
            task_id="sample/3",
            prompt="from typing import List\n\n\ndef first(xs: List[int]):\n",
            test="def check(candidate):\n    assert candidate([3, 4]) == 3",
            entry_point="first",
        )

        result = check_correctness(
            problem,
            "def first(xs: List[int]):\n    return xs[0]",
            timeout_seconds=2,
        )

        self.assertTrue(result.passed, result.stderr)


class DatasetTests(unittest.TestCase):
    def test_humaneval_loader_uses_current_dataset_id(self) -> None:
        row = {
            "task_id": "HumanEval/0",
            "prompt": "def f():\n",
            "test": "def check(candidate):\n    assert candidate() == 1",
            "entry_point": "f",
        }

        with patch("datasets.load_dataset", return_value=[row]) as load_dataset:
            problems = load_humaneval(limit=1)

        load_dataset.assert_called_once_with(DEFAULT_HUMANEVAL_DATASET, split="test")
        self.assertEqual(problems[0].task_id, "HumanEval/0")


class RunProblemTests(unittest.TestCase):
    def test_run_problem_early_stop_keeps_problem_local_seed_block(self) -> None:
        class FakeGenerator:
            def generate(self, humaneval_prompt: str, seed: int):
                class Sample:
                    text = "    return 1"
                    generated_tokens = 1
                    generation_seconds = 0.1

                seen_seeds.append(seed)
                return Sample()

        problem = CodingProblem(
            task_id="sample/seed",
            prompt="def f():\n",
            test="def check(candidate):\n    assert True",
            entry_point="f",
        )
        seen_seeds: list[int] = []

        record = run_problem(
            problem=problem,
            generator=FakeGenerator(),
            n=5,
            seed=100,
            global_attempt_start=10,
            timeout_seconds=2,
            early_stop=True,
        )

        self.assertEqual(seen_seeds, [110])
        self.assertEqual(record["first_success_attempt"], 1)


class MetricsTests(unittest.TestCase):
    def test_pass_at_k_estimator(self) -> None:
        self.assertEqual(estimate_pass_at_k(10, 0, 1), 0.0)
        self.assertEqual(estimate_pass_at_k(10, 10, 5), 1.0)
        self.assertAlmostEqual(estimate_pass_at_k(10, 1, 1), 0.1)

    def test_summary_core_metrics(self) -> None:
        records = [
            {
                "solved": True,
                "first_success_attempt": 2,
                "problem_wall_seconds": 3.0,
                "attempts": [
                    {
                        "passed": False,
                        "generated_tokens": 10,
                        "generation_seconds": 1.0,
                        "verification_seconds": 0.1,
                    },
                    {
                        "passed": True,
                        "generated_tokens": 20,
                        "generation_seconds": 2.0,
                        "verification_seconds": 0.1,
                    },
                ],
            },
            {
                "solved": False,
                "first_success_attempt": None,
                "problem_wall_seconds": 1.0,
                "attempts": [
                    {
                        "passed": False,
                        "generated_tokens": 5,
                        "generation_seconds": 0.5,
                        "verification_seconds": 0.1,
                    },
                    {
                        "passed": False,
                        "generated_tokens": 5,
                        "generation_seconds": 0.5,
                        "verification_seconds": 0.1,
                    },
                ],
            },
        ]

        summary = summarize(records, requested_n=2)

        self.assertEqual(summary["num_tasks"], 2)
        self.assertFalse(summary["adaptive_sampling_used"])
        self.assertEqual(summary["strict_pass_at_1"], 0.0)
        self.assertEqual(summary["observed_best_of_n"], 0.5)
        self.assertEqual(summary["mean_attempts_until_success_or_budget"], 2)
        self.assertEqual(summary["total_tokens_until_success_or_budget"], 40)
        self.assertAlmostEqual(summary["token_savings_if_oracle_early_stop"], 0.0)


class ReportingTests(unittest.TestCase):
    def test_write_run_artifacts(self) -> None:
        import shutil
        from pathlib import Path

        records = [
            {
                "task_id": "HumanEval/0",
                "entry_point": "f",
                "solved": True,
                "first_success_attempt": 1,
                "problem_wall_seconds": 1.0,
                "attempts": [
                    {
                        "attempt": 1,
                        "seed": 123,
                        "passed": True,
                        "generated_tokens": 3,
                        "generation_seconds": 0.5,
                        "verification_seconds": 0.1,
                        "error_type": None,
                    }
                ],
            }
        ]
        summary = summarize(records, requested_n=1)

        temp_dir = Path("runs/tmp_tests/reporting")
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            paths = write_run_artifacts(records, summary, temp_dir / "run")

            self.assertTrue(Path(paths["report_md"]).exists())
            self.assertTrue(Path(paths["tasks_csv"]).exists())
            self.assertTrue(Path(paths["attempts_csv"]).exists())
            self.assertTrue(Path(paths["curve_csv"]).exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class AnalysisTests(unittest.TestCase):
    def _sample_run(self) -> RunBundle:
        records = [
            {
                "task_id": "HumanEval/0",
                "entry_point": "f",
                "solved": True,
                "first_success_attempt": 2,
                "problem_wall_seconds": 1.5,
                "attempts": [
                    {
                        "attempt": 1,
                        "seed": 10,
                        "passed": False,
                        "generated_tokens": 10,
                        "generation_seconds": 1.0,
                        "verification_seconds": 0.1,
                        "error_type": "AssertionError",
                        "raw_completion": "return 0",
                        "candidate_source": "def f():\n    return 0",
                    },
                    {
                        "attempt": 2,
                        "seed": 11,
                        "passed": True,
                        "generated_tokens": 20,
                        "generation_seconds": 2.0,
                        "verification_seconds": 0.1,
                        "error_type": None,
                        "raw_completion": "return 1",
                        "candidate_source": "def f():\n    return 1",
                    },
                ],
            },
            {
                "task_id": "HumanEval/1",
                "entry_point": "g",
                "solved": False,
                "first_success_attempt": None,
                "problem_wall_seconds": 1.0,
                "attempts": [
                    {
                        "attempt": 1,
                        "seed": 12,
                        "passed": False,
                        "generated_tokens": 5,
                        "generation_seconds": 0.5,
                        "verification_seconds": 0.1,
                        "error_type": "AssertionError",
                        "raw_completion": "return 0",
                        "candidate_source": "def g():\n    return 0",
                    }
                ],
            },
        ]
        return RunBundle(label="sample", records=records, summary=summarize(records, requested_n=2))

    def test_attempt_and_task_frames_keep_core_fields(self) -> None:
        run = self._sample_run()

        attempts = attempts_frame(run)
        tasks = tasks_frame(run)

        self.assertEqual(len(attempts), 3)
        self.assertEqual(attempts["passed"].sum(), 1)
        self.assertAlmostEqual(attempts.loc[0, "tokens_per_second"], 10.0)
        self.assertEqual(tasks.loc[tasks["task_id"] == "HumanEval/0", "difficulty"].item(), "fragile")
        self.assertEqual(tasks.loc[tasks["task_id"] == "HumanEval/1", "difficulty"].item(), "unresolved")

    def test_counterfactual_early_stop_counts_saved_work(self) -> None:
        savings = counterfactual_early_stop_frame(self._sample_run())

        solved_row = savings.loc[savings["task_id"] == "HumanEval/0"].iloc[0]
        unresolved_row = savings.loc[savings["task_id"] == "HumanEval/1"].iloc[0]

        self.assertEqual(solved_row["effective_attempts"], 2)
        self.assertEqual(solved_row["saved_tokens"], 0)
        self.assertEqual(unresolved_row["effective_attempts"], 1)

    def test_diversity_and_bootstrap_tables_are_not_empty(self) -> None:
        run = self._sample_run()

        diversity = diversity_frame(run)
        confidence = bootstrap_success_table([run])

        self.assertIn("mean_pairwise_code_distance", diversity.columns)
        self.assertEqual(len(confidence), 2)
        self.assertTrue(confidence["mean"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
