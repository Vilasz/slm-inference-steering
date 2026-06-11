from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from scripts.generate_final_report_artifacts import generate_artifacts


class FinalReportingTests(unittest.TestCase):
    def test_generate_artifacts_without_runs(self) -> None:
        root = Path("runs/tmp_tests/final_reporting_empty")
        shutil.rmtree(root, ignore_errors=True)
        try:
            summary = generate_artifacts(root)

            self.assertTrue((root / "runs/final_summary.json").exists())
            self.assertTrue((root / "reports/tables/main_results.csv").exists())
            self.assertIn("tables", summary)
            self.assertIn("figures", summary)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_generate_artifacts_collects_phase4_summary(self) -> None:
        root = Path("runs/tmp_tests/final_reporting_with_data")
        shutil.rmtree(root, ignore_errors=True)
        phase4 = root / "runs/phase4"
        phase4.mkdir(parents=True, exist_ok=True)
        summary_path = phase4 / "layer12_correctness_direction_alpha1_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "num_tasks": 2,
                    "requested_n": 2,
                    "observed_best_of_n": 0.5,
                    "mean_tokens_until_success_or_budget": 123.0,
                    "steering": {
                        "model_id": "synthetic",
                        "benchmark": "humaneval_stress",
                        "layer": 12,
                        "alpha": 1.0,
                        "direction_type": "correctness_direction",
                    },
                }
            ),
            encoding="utf-8",
        )
        try:
            summary = generate_artifacts(root)

            self.assertEqual(summary["num_result_rows"], 1)
            self.assertIn("phase4", summary["available_phases"])
            self.assertTrue((root / "reports/tables/steering_controls.csv").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
