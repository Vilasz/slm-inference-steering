from __future__ import annotations

from dataclasses import dataclass

from slm_steering.datasets import CodingProblem


@dataclass(frozen=True)
class StressProblem:
    task_id: str
    prompt: str
    test: str
    entry_point: str
    benchmark_name: str
    difficulty_tags: tuple[str, ...]

    def to_coding_problem(self) -> CodingProblem:
        return CodingProblem(
            task_id=self.task_id,
            prompt=self.prompt,
            test=self.test,
            entry_point=self.entry_point,
        )


def load_humaneval_stress(limit: int | None = None, offset: int = 0) -> list[CodingProblem]:
    problems = [
        StressProblem(
            task_id="HumanEvalStress/0",
            prompt=(
                "from typing import List\n\n"
                "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
                "    \"\"\"Return True if any two distinct numbers are closer than threshold.\n"
                "    Handle empty lists, duplicates, negative values and very small thresholds.\n"
                "    \"\"\"\n"
            ),
            test=(
                "def check(candidate):\n"
                "    assert candidate([], 0.1) is False\n"
                "    assert candidate([1.0], 0.1) is False\n"
                "    assert candidate([1.0, 1.0], 0.0) is False\n"
                "    assert candidate([1.0, 1.0], 0.001) is True\n"
                "    assert candidate([-1.0, -1.05, 3.0], 0.1) is True\n"
            ),
            entry_point="has_close_elements",
            benchmark_name="humaneval_stress",
            difficulty_tags=("edge_cases", "duplicates", "negative_numbers"),
        ),
        StressProblem(
            task_id="HumanEvalStress/1",
            prompt=(
                "def truncate_number(number: float) -> float:\n"
                "    \"\"\"Return the fractional part of a number.\n"
                "    Preserve behavior for negative and whole numbers.\n"
                "    \"\"\"\n"
            ),
            test=(
                "def check(candidate):\n"
                "    assert abs(candidate(3.5) - 0.5) < 1e-6\n"
                "    assert abs(candidate(10.0) - 0.0) < 1e-6\n"
                "    assert abs(candidate(-2.75) - 0.25) < 1e-6\n"
            ),
            entry_point="truncate_number",
            benchmark_name="humaneval_stress",
            difficulty_tags=("negative_numbers", "numeric_precision"),
        ),
        StressProblem(
            task_id="HumanEvalStress/2",
            prompt=(
                "from typing import List\n\n"
                "def separate_paren_groups(paren_string: str) -> List[str]:\n"
                "    \"\"\"Split balanced parenthesis groups and ignore surrounding spaces.\n"
                "    Return an empty list for an empty or whitespace-only string.\n"
                "    \"\"\"\n"
            ),
            test=(
                "def check(candidate):\n"
                "    assert candidate('') == []\n"
                "    assert candidate('   ') == []\n"
                "    assert candidate('(()) ()') == ['(())', '()']\n"
                "    assert candidate(' (()()) ((())) ') == ['(()())', '((()))']\n"
            ),
            entry_point="separate_paren_groups",
            benchmark_name="humaneval_stress",
            difficulty_tags=("strings", "empty_inputs", "whitespace"),
        ),
    ]
    selected = problems[offset:]
    if limit is not None:
        selected = selected[:limit]
    return [problem.to_coding_problem() for problem in selected]
