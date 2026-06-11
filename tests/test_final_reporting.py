from __future__ import annotations

import json
import shutil
import unittest
import csv
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

    def test_generate_artifacts_collects_phase2_metadata_and_zero_metrics(self) -> None:
        root = Path("runs/tmp_tests/final_reporting_phase2")
        shutil.rmtree(root, ignore_errors=True)
        phase2 = root / "runs/phase2"
        phase2.mkdir(parents=True, exist_ok=True)
        summary_path = phase2 / "qwen05b_temp02_n1_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "num_tasks": 2,
                    "requested_n": 1,
                    "observed_best_of_n": 0.0,
                    "mean_tokens_until_success_or_budget": 0.0,
                    "config": {
                        "model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                        "benchmark": "humaneval",
                    },
                    "phase2": {
                        "model_key": "qwen2.5-coder-0.5b-instruct",
                        "model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                        "benchmark": "humaneval",
                        "temperature": 0.2,
                    },
                }
            ),
            encoding="utf-8",
        )
        try:
            summary = generate_artifacts(root)

            self.assertEqual(summary["num_result_rows"], 1)
            rows = read_csv(root / "reports/tables/main_results.csv")
            self.assertEqual(rows[0]["model_id"], "Qwen/Qwen2.5-Coder-0.5B-Instruct")
            self.assertEqual(rows[0]["benchmark"], "humaneval")
            self.assertEqual(rows[0]["observed_best_of_n"], "0.0")
            self.assertEqual(rows[0]["mean_tokens_until_success_or_budget"], "0.0")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_generate_artifacts_collects_phase35_summary_and_model_matrix(self) -> None:
        root = Path("runs/tmp_tests/final_reporting_phase35_matrix")
        shutil.rmtree(root, ignore_errors=True)
        phase35 = root / "runs/phase35/latent_geometry"
        phase35.mkdir(parents=True, exist_ok=True)
        (phase35 / "summary.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "model_id": "synthetic-latent-model",
                    "num_tasks": 3,
                    "num_activation_samples": 9,
                    "best_probe_auc": 0.75,
                    "permutation_p_value": 0.04,
                    "latent_score_delta_auc": 0.08,
                    "recommended_steering_layers": [8, 12],
                }
            ),
            encoding="utf-8",
        )
        matrix = root / "runs/model_matrix"
        matrix.mkdir(parents=True, exist_ok=True)
        (matrix / "deepseek_summary.json").write_text(
            json.dumps(
                {
                    "num_tasks": 2,
                    "requested_n": 1,
                    "observed_best_of_n": 1.0,
                    "mean_tokens_until_success_or_budget": 42.0,
                    "config": {"dataset": "openai/openai_humaneval/test"},
                    "matrix": {
                        "model_key": "deepseek-coder-1.3b-instruct",
                        "model_id": "deepseek-ai/deepseek-coder-1.3b-instruct",
                        "family": "DeepSeek-Coder",
                        "decoding_key": "greedy_n1",
                        "temperature": 0.0,
                        "early_stop": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        try:
            summary = generate_artifacts(root)

            self.assertEqual(summary["num_result_rows"], 2)
            self.assertIn("phase35", summary["available_phases"])
            self.assertIn("model_matrix", summary["available_phases"])
            stats = read_csv(root / "reports/tables/statistical_tests.csv")
            self.assertIn("best_probe_auc", {row["metric"] for row in stats})
            cross_model = read_csv(root / "reports/tables/cross_model_results.csv")
            self.assertEqual(cross_model[0]["model_key"], "deepseek-coder-1.3b-instruct")
            self.assertEqual(cross_model[0]["benchmark"], "humaneval")
        finally:
            shutil.rmtree(root, ignore_errors=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


if __name__ == "__main__":
    unittest.main()
