from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from slm_steering.experiments.runner import load_experiment_config, run_experiment_config


class ExperimentRunnerTests(unittest.TestCase):
    def test_load_json_compatible_yaml_config(self) -> None:
        path = Path("runs/tmp_tests/runner_config.yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"name": "sample", "steps": []}', encoding="utf-8")
        try:
            config = load_experiment_config(path)

            self.assertEqual(config["name"], "sample")
        finally:
            path.unlink(missing_ok=True)

    def test_smoke_runner_writes_artifacts(self) -> None:
        root = Path("runs/tmp_tests/runner")
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        config_path = root / "smoke.yaml"
        config_path.write_text(
            json.dumps(
                {
                    "name": "smoke",
                    "output_dir": "runs/smoke",
                    "steps": [
                        {
                            "name": "benchmarks",
                            "type": "benchmark_smoke",
                            "benchmarks": ["humaneval_stress"],
                            "limit": 1,
                        },
                        {"name": "report", "type": "report"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        try:
            summary = run_experiment_config(config_path, root=root)

            self.assertTrue(Path(summary["summary_path"]).exists())
            self.assertTrue((root / "runs/smoke/benchmarks.json").exists())
            self.assertTrue((root / "reports/tables/main_results.csv").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

