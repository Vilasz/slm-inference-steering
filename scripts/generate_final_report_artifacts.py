from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PHASE_DIRS = {
    "phase1": Path("runs/phase1_pilot"),
    "phase2": Path("runs/phase2"),
    "phase3": Path("runs/phase3"),
    "phase35": Path("runs/phase35"),
    "phase4": Path("runs/phase4"),
    "phase5": Path("runs/phase5"),
}


def main() -> None:
    artifacts = generate_artifacts(ROOT)
    print(json.dumps(artifacts, indent=2, ensure_ascii=False))


def generate_artifacts(root: Path = ROOT) -> dict[str, Any]:
    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)

    summary_records = collect_summary_records(root)
    summary_frame = pd.DataFrame(summary_records)
    if summary_frame.empty:
        summary_frame = empty_summary_frame()

    main_results_path = tables_dir / "main_results.csv"
    summary_frame.to_csv(main_results_path, index=False)

    steering_controls = steering_controls_frame(summary_frame)
    steering_controls_path = tables_dir / "steering_controls.csv"
    steering_controls.to_csv(steering_controls_path, index=False)

    adaptive = adaptive_policy_frame(summary_frame)
    adaptive_path = tables_dir / "adaptive_policy_results.csv"
    adaptive.to_csv(adaptive_path, index=False)

    robustness = robustness_frame(summary_frame)
    robustness_path = tables_dir / "robustness_results.csv"
    robustness.to_csv(robustness_path, index=False)

    cross_model = cross_model_frame(summary_frame)
    cross_model_path = tables_dir / "cross_model_results.csv"
    cross_model.to_csv(cross_model_path, index=False)

    stats = statistical_tests_frame(summary_frame)
    stats_path = tables_dir / "statistical_tests.csv"
    stats.to_csv(stats_path, index=False)

    figure_paths = write_figures(summary_frame, figures_dir)
    final_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_result_rows": int(len(summary_frame)),
        "available_phases": sorted(set(summary_frame["phase"].dropna().astype(str))),
        "best_accuracy_row": best_row(summary_frame, "observed_best_of_n"),
        "best_cost_row": best_row(summary_frame, "mean_tokens_until_success_or_budget", ascending=True),
        "tables": {
            "main_results": str(main_results_path),
            "steering_controls": str(steering_controls_path),
            "adaptive_policy_results": str(adaptive_path),
            "robustness_results": str(robustness_path),
            "cross_model_results": str(cross_model_path),
            "statistical_tests": str(stats_path),
        },
        "figures": {path.stem: str(path) for path in figure_paths},
        "notes": [
            "Tabelas podem estar vazias quando a fase correspondente ainda nao foi executada.",
            "Use os notebooks para interpretar resultados; este script apenas consolida artefatos.",
        ],
    }
    final_summary_path = root / "runs" / "final_summary.json"
    final_summary_path.write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return final_summary


def collect_summary_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for phase, relative_dir in PHASE_DIRS.items():
        phase_dir = root / relative_dir
        for path in sorted(phase_dir.rglob("*_summary.json")) if phase_dir.exists() else []:
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            row = flatten_summary(summary)
            row["phase"] = phase
            row["summary_path"] = str(path)
            row["run_label"] = path.stem.replace("_summary", "")
            records.append(row)
    return records


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metadata = summary.get("steering") or summary.get("policy") or summary.get("run") or {}
    return {
        "model_id": metadata.get("model_id") or summary.get("model_id"),
        "benchmark": metadata.get("benchmark") or summary.get("benchmark"),
        "policy": metadata.get("name"),
        "layer": metadata.get("layer"),
        "alpha": metadata.get("alpha"),
        "direction_type": metadata.get("direction_type"),
        "num_tasks": summary.get("num_tasks"),
        "requested_n": summary.get("requested_n"),
        "total_attempts": summary.get("total_attempts"),
        "strict_pass_at_1": summary.get("strict_pass_at_1"),
        "observed_best_of_n": summary.get("observed_best_of_n") or summary.get("success_rate"),
        "mean_generated_tokens_per_attempt": summary.get("mean_generated_tokens_per_attempt"),
        "mean_tokens_until_success_or_budget": summary.get("mean_tokens_until_success_or_budget")
        or summary.get("mean_tokens"),
        "mean_attempts_until_success_or_budget": summary.get("mean_attempts_until_success_or_budget")
        or summary.get("mean_attempts"),
        "diversity_score": summary.get("diversity_score"),
        "collapse_score": summary.get("collapse_score"),
        "syntax_error_rate": summary.get("syntax_error_rate"),
        "runtime_error_rate": summary.get("runtime_error_rate"),
        "test_failure_rate": summary.get("test_failure_rate"),
        "budget_saved_vs_fixed_n": summary.get("budget_saved_vs_fixed_n"),
        "tokens_per_success": summary.get("tokens_per_success"),
    }


def empty_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "phase",
            "run_label",
            "summary_path",
            "model_id",
            "benchmark",
            "policy",
            "layer",
            "alpha",
            "direction_type",
            "num_tasks",
            "requested_n",
            "total_attempts",
            "strict_pass_at_1",
            "observed_best_of_n",
            "mean_generated_tokens_per_attempt",
            "mean_tokens_until_success_or_budget",
            "mean_attempts_until_success_or_budget",
            "diversity_score",
            "collapse_score",
            "syntax_error_rate",
            "runtime_error_rate",
            "test_failure_rate",
            "budget_saved_vs_fixed_n",
            "tokens_per_success",
        ]
    )


def steering_controls_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run_label",
        "layer",
        "alpha",
        "direction_type",
        "observed_best_of_n",
        "mean_tokens_until_success_or_budget",
        "diversity_score",
        "collapse_score",
    ]
    if frame.empty or "phase" not in frame:
        return pd.DataFrame(columns=columns)
    return frame.loc[frame["phase"].eq("phase4"), columns].copy()


def adaptive_policy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run_label",
        "policy",
        "observed_best_of_n",
        "mean_attempts_until_success_or_budget",
        "mean_tokens_until_success_or_budget",
        "budget_saved_vs_fixed_n",
        "tokens_per_success",
    ]
    if frame.empty or "phase" not in frame:
        return pd.DataFrame(columns=columns)
    return frame.loc[frame["phase"].eq("phase5"), columns].copy()


def robustness_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["run_label", "benchmark", "observed_best_of_n", "runtime_error_rate", "test_failure_rate"]
    if frame.empty or "benchmark" not in frame:
        return pd.DataFrame(columns=columns)
    mask = frame["benchmark"].astype(str).str.contains("stress", case=False, na=False)
    return frame.loc[mask, columns].copy()


def cross_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["run_label", "model_id", "benchmark", "observed_best_of_n", "mean_tokens_until_success_or_budget"]
    if frame.empty or "model_id" not in frame:
        return pd.DataFrame(columns=columns)
    return frame.loc[frame["model_id"].notna(), columns].copy()


def statistical_tests_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["comparison", "metric", "effect_size", "note"]
    if frame.empty or len(frame) < 2:
        return pd.DataFrame(
            [{"comparison": "pending", "metric": "observed_best_of_n", "effect_size": None, "note": "Execute ao menos duas runs."}],
            columns=columns,
        )
    metric = pd.to_numeric(frame["observed_best_of_n"], errors="coerce")
    effect = float(metric.max() - metric.min()) if metric.notna().any() else None
    return pd.DataFrame(
        [
            {
                "comparison": "max_minus_min",
                "metric": "observed_best_of_n",
                "effect_size": effect,
                "note": "Efeito descritivo; substitua por teste pareado quando houver mesmas tarefas por run.",
            }
        ],
        columns=columns,
    )


def write_figures(frame: pd.DataFrame, figures_dir: Path) -> list[Path]:
    paths = [
        write_accuracy_cost_plot(frame, figures_dir / "accuracy_cost_scatter.png"),
        write_phase_coverage_plot(frame, figures_dir / "phase_coverage.png"),
    ]
    return paths


def write_accuracy_cost_plot(frame: pd.DataFrame, path: Path) -> Path:
    plt.figure(figsize=(7, 4.5))
    x = (
        pd.to_numeric(frame["mean_tokens_until_success_or_budget"], errors="coerce")
        if "mean_tokens_until_success_or_budget" in frame.columns
        else pd.Series(dtype=float)
    )
    y = (
        pd.to_numeric(frame["observed_best_of_n"], errors="coerce")
        if "observed_best_of_n" in frame.columns
        else pd.Series(dtype=float)
    )
    if x.notna().any() and y.notna().any():
        for phase, group in frame.assign(_x=x, _y=y).dropna(subset=["_x", "_y"]).groupby("phase"):
            plt.scatter(group["_x"], group["_y"], label=phase, s=50)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Sem resultados suficientes", ha="center", va="center")
    plt.xlabel("Tokens ate sucesso ou orcamento")
    plt.ylabel("Acuracia Best-of-N observada")
    plt.title("Fronteira custo-acuracia")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def write_phase_coverage_plot(frame: pd.DataFrame, path: Path) -> Path:
    plt.figure(figsize=(7, 4.5))
    if "phase" in frame.columns and not frame.empty:
        counts = frame["phase"].value_counts().sort_index()
        plt.bar(counts.index.astype(str), counts.values)
    else:
        plt.text(0.5, 0.5, "Sem resultados suficientes", ha="center", va="center")
    plt.xlabel("Fase")
    plt.ylabel("Runs consolidadas")
    plt.title("Cobertura experimental")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def best_row(frame: pd.DataFrame, column: str, *, ascending: bool = False) -> dict[str, Any] | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    if not values.notna().any():
        return None
    index = values.sort_values(ascending=ascending).index[0]
    row = frame.loc[index].to_dict()
    return {key: value for key, value in row.items() if pd.notna(value)}


if __name__ == "__main__":
    main()
