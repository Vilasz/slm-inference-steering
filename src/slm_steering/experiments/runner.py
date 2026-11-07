from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_steering.benchmark_registry import load_benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class StepResult:
    name: str
    type: str
    status: str
    artifact: str | None = None
    returncode: int | None = None
    command: list[str] | None = None


def load_experiment_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise ValueError(
                f"{path} nao e JSON valido. Instale PyYAML para usar YAML completo "
                "ou mantenha o arquivo YAML como JSON valido."
            ) from exc
        return yaml.safe_load(text)


def run_experiment_config(
    config_path: Path,
    *,
    root: Path,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    name = str(config.get("name") or config_path.stem)
    output_dir = root / str(config.get("output_dir") or f"runs/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for step in config.get("steps", []):
        result = run_step(step, root=root, output_dir=output_dir, dry_run=dry_run)
        results.append(result)
        if result.status == "failed" and not continue_on_error:
            break

    summary = {
        "name": name,
        "config_path": str(config_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "steps": [result.__dict__ for result in results],
    }
    summary_path = output_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def run_step(step: dict[str, Any], *, root: Path, output_dir: Path, dry_run: bool) -> StepResult:
    step_type = str(step.get("type") or "command")
    name = str(step.get("name") or step_type)
    if step_type == "benchmark_smoke":
        return run_benchmark_smoke(step, name=name, output_dir=output_dir, dry_run=dry_run)
    if step_type == "report":
        return run_report_step(name=name, root=root, dry_run=dry_run)
    if step_type == "command":
        return run_command_step(step, name=name, root=root, dry_run=dry_run)
    raise ValueError(f"Tipo de step desconhecido: {step_type}")


def run_benchmark_smoke(
    step: dict[str, Any],
    *,
    name: str,
    output_dir: Path,
    dry_run: bool,
) -> StepResult:
    artifact_path = output_dir / f"{name}.json"
    benchmarks = [str(item) for item in step.get("benchmarks", ["humaneval_stress"])]
    limit = int(step.get("limit", 1))
    if dry_run:
        return StepResult(name=name, type="benchmark_smoke", status="planned", artifact=str(artifact_path))

    rows = []
    for benchmark in benchmarks:
        problems = load_benchmark(benchmark, limit=limit)
        rows.append(
            {
                "benchmark": benchmark,
                "num_examples": len(problems),
                "first_task_id": problems[0].task_id if problems else None,
                "first_entry_point": problems[0].entry_point if problems else None,
            }
        )
    artifact_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return StepResult(name=name, type="benchmark_smoke", status="completed", artifact=str(artifact_path))


def run_report_step(*, name: str, root: Path, dry_run: bool) -> StepResult:
    if dry_run:
        return StepResult(name=name, type="report", status="planned")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.generate_final_report_artifacts import generate_artifacts

    summary = generate_artifacts(root)
    return StepResult(
        name=name,
        type="report",
        status="completed",
        artifact=str(root / "runs" / "final_summary.json"),
        returncode=int(summary.get("num_result_rows", 0)),
    )


def run_command_step(step: dict[str, Any], *, name: str, root: Path, dry_run: bool) -> StepResult:
    command = normalize_command(step.get("command"))
    if dry_run:
        return StepResult(name=name, type="command", status="planned", command=command)
    completed = subprocess.run(command, cwd=root, check=False)
    return StepResult(
        name=name,
        type="command",
        status="completed" if completed.returncode == 0 else "failed",
        returncode=completed.returncode,
        command=command,
    )


def normalize_command(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("Steps do tipo command precisam de uma lista em 'command'.")
    command = [str(item) for item in value]
    return [sys.executable if item == "{python}" else item for item in command]
