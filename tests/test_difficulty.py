from __future__ import annotations

import unittest

from slm_steering.difficulty import classify_task_difficulty, difficulty_frame


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


class DifficultyTests(unittest.TestCase):
    def test_hard_when_no_attempt_passes(self) -> None:
        row = classify_task_difficulty(make_record("hard", [False, False, False]))

        self.assertEqual(row["difficulty"], "hard")
        self.assertFalse(row["solved_by_sampling"])

    def test_easy_when_first_attempt_passes(self) -> None:
        row = classify_task_difficulty(make_record("easy", [True, False, False]))

        self.assertEqual(row["difficulty"], "easy")
        self.assertTrue(row["pass_at_1"])

    def test_fragile_when_only_one_late_attempt_passes(self) -> None:
        row = classify_task_difficulty(make_record("fragile", [False, False, True, False, False]))

        self.assertEqual(row["difficulty"], "fragile")
        self.assertTrue(row["solved_by_sampling"])
        self.assertAlmostEqual(row["success_rate"], 0.2)

    def test_sampling_sensitive_when_best_of_n_is_stable_enough(self) -> None:
        row = classify_task_difficulty(make_record("sampling", [False, True, False, True, False]))

        self.assertEqual(row["difficulty"], "sampling_sensitive")
        self.assertTrue(row["solved_by_sampling"])

    def test_frame_keeps_one_row_per_task(self) -> None:
        frame = difficulty_frame(
            [
                make_record("a", [True]),
                make_record("b", [False, False]),
            ],
            run_label="run",
        )

        self.assertEqual(len(frame), 2)
        self.assertEqual(set(frame["run"]), {"run"})


if __name__ == "__main__":
    unittest.main()
