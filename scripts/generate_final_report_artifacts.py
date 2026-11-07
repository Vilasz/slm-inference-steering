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
    "model_matrix": Path("runs/model_matrix"),
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

    paired = paired_comparisons_frame(root, summary_frame)
    paired_path = tables_dir / "paired_comparisons.csv"
    paired.to_csv(paired_path, index=False)

    figure_paths = write_figures(summary_frame, figures_dir, root)
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
            "paired_comparisons": str(paired_path),
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
        for path in summary_paths(phase_dir, phase):
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            row = flatten_summary(summary)
            row["phase"] = phase
            row["summary_path"] = str(path)
            row["run_label"] = run_label_from_summary_path(path)
            records.append(row)
    return records


def summary_paths(phase_dir: Path, phase: str) -> list[Path]:
    if not phase_dir.exists():
        return []
    paths = set(phase_dir.rglob("*_summary.json"))
    if phase == "phase35":
        paths.update(phase_dir.rglob("summary.json"))
    return sorted(paths)


def run_label_from_summary_path(path: Path) -> str:
    if path.stem == "summary":
        return path.parent.name
    return path.stem.replace("_summary", "")


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metadata = merged_metadata(summary)
    benchmark = metadata.get("benchmark") or summary.get("benchmark") or infer_benchmark(metadata)
    return {
        "model_id": metadata.get("model_id") or summary.get("model_id"),
        "model_key": metadata.get("model_key"),
        "family": metadata.get("family"),
        "parameters_b": metadata.get("parameters_b"),
        "benchmark": benchmark,
        "policy": metadata.get("name"),
        "layer": metadata.get("layer"),
        "alpha": metadata.get("alpha"),
        "direction_type": metadata.get("direction_type"),
        "decoding_key": metadata.get("decoding_key"),
        "temperature": metadata.get("temperature"),
        "top_p": metadata.get("top_p"),
        "early_stop": metadata.get("early_stop"),
        "status": summary.get("status"),
        "num_tasks": summary.get("num_tasks"),
        "num_activation_samples": summary.get("num_activation_samples"),
        "requested_n": summary.get("requested_n"),
        "total_attempts": summary.get("total_attempts"),
        "strict_pass_at_1": summary.get("strict_pass_at_1"),
        "observed_best_of_n": first_present(summary, "observed_best_of_n", "success_rate"),
        "mean_generated_tokens_per_attempt": summary.get("mean_generated_tokens_per_attempt"),
        "mean_tokens_until_success_or_budget": first_present(
            summary,
            "mean_tokens_until_success_or_budget",
            "mean_tokens",
        ),
        "mean_attempts_until_success_or_budget": first_present(
            summary,
            "mean_attempts_until_success_or_budget",
            "mean_attempts",
        ),
        "diversity_score": summary.get("diversity_score"),
        "collapse_score": summary.get("collapse_score"),
        "syntax_error_rate": summary.get("syntax_error_rate"),
        "runtime_error_rate": summary.get("runtime_error_rate"),
        "test_failure_rate": summary.get("test_failure_rate"),
        "budget_saved_vs_fixed_n": summary.get("budget_saved_vs_fixed_n"),
        "tokens_per_success": summary.get("tokens_per_success"),
        "best_probe_layer": summary.get("best_probe_layer"),
        "best_probe_auc": summary.get("best_probe_auc"),
        "permutation_p_value": summary.get("permutation_p_value"),
        "latent_score_delta_auc": summary.get("latent_score_delta_auc"),
        "mean_attempt_reduction_candidate": summary.get("mean_attempt_reduction_candidate"),
        "recommended_steering_layers": json.dumps(
            summary.get("recommended_steering_layers") or [],
            ensure_ascii=False,
        ),
    }


def merged_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for section in ["config", "phase2", "matrix", "run", "policy", "steering"]:
        values = summary.get(section)
        if isinstance(values, dict):
            metadata.update(values)
    return metadata


def first_present(values: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = values.get(key)
        if value is not None:
            return value
    return None


def infer_benchmark(metadata: dict[str, Any]) -> str | None:
    dataset = str(metadata.get("dataset") or metadata.get("dataset_id") or "").lower()
    if "humaneval" in dataset:
        return "humaneval"
    return None


def empty_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "phase",
            "run_label",
            "summary_path",
            "model_id",
            "model_key",
            "family",
            "parameters_b",
            "benchmark",
            "policy",
            "layer",
            "alpha",
            "direction_type",
            "decoding_key",
            "temperature",
            "top_p",
            "early_stop",
            "status",
            "num_tasks",
            "num_activation_samples",
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
            "best_probe_layer",
            "best_probe_auc",
            "permutation_p_value",
            "latent_score_delta_auc",
            "mean_attempt_reduction_candidate",
            "recommended_steering_layers",
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
    columns = [
        "run_label",
        "phase",
        "model_id",
        "model_key",
        "family",
        "benchmark",
        "decoding_key",
        "requested_n",
        "temperature",
        "early_stop",
        "observed_best_of_n",
        "mean_tokens_until_success_or_budget",
    ]
    if frame.empty or "model_id" not in frame:
        return pd.DataFrame(columns=columns)
    mask = frame["model_id"].notna() & frame["observed_best_of_n"].notna()
    return frame.loc[mask, columns].copy()


def statistical_tests_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["comparison", "metric", "effect_size", "note"]
    rows = phase35_statistical_rows(frame)
    if len(frame) >= 2:
        metric = pd.to_numeric(frame["observed_best_of_n"], errors="coerce")
        effect = float(metric.max() - metric.min()) if metric.notna().any() else None
        rows.append(
            {
                "comparison": "max_minus_min",
                "metric": "observed_best_of_n",
                "effect_size": effect,
                "note": "Efeito descritivo; substitua por teste pareado quando houver mesmas tarefas por run.",
            }
        )
    if rows:
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(
        [{"comparison": "pending", "metric": "observed_best_of_n", "effect_size": None, "note": "Execute ao menos duas runs."}],
        columns=columns,
    )


def phase35_statistical_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty or "phase" not in frame:
        return []
    rows = []
    phase35 = frame[frame["phase"].eq("phase35")]
    if phase35.empty:
        return rows
    for _, row in phase35.iterrows():
        status = row.get("status")
        run_label = row.get("run_label")
        for metric, note in [
            ("best_probe_auc", "AUC do melhor probe linear por camada."),
            ("permutation_p_value", "Significancia empirica da direcao de corretude."),
            ("latent_score_delta_auc", "Ganho incremental da regressao ao incluir score latente."),
            ("mean_attempt_reduction_candidate", "Reducao media candidata em tentativas ate sucesso."),
        ]:
            value = row.get(metric)
            if pd.notna(value):
                rows.append(
                    {
                        "comparison": f"{run_label}:{status or 'summary'}",
                        "metric": metric,
                        "effect_size": value,
                        "note": note,
                    }
                )
    return rows


def paired_comparisons_frame(root: Path, frame: pd.DataFrame, *, max_pairs: int = 100) -> pd.DataFrame:
    columns = [
        "left_run",
        "right_run",
        "benchmark",
        "n_tasks",
        "accuracy_delta",
        "accuracy_ci_low",
        "accuracy_ci_high",
        "accuracy_p_value",
        "token_delta",
        "token_ci_low",
        "token_ci_high",
        "dominance",
    ]
    if frame.empty or "summary_path" not in frame:
        return pd.DataFrame(columns=columns)

    records_by_label = {}
    for _, row in frame.iterrows():
        jsonl_path = jsonl_path_for_summary(root, row.get("summary_path"))
        if jsonl_path is None or not jsonl_path.exists():
            continue
        try:
            from slm_steering.reporting import read_jsonl

            records = read_jsonl(jsonl_path)
        except Exception:
            continue
        if records:
            records_by_label[str(row["run_label"])] = {
                "records": records,
                "benchmark": row.get("benchmark") or "unknown",
            }

    labels = sorted(records_by_label)
    rows = []
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1 :]:
            left = records_by_label[left_label]
            right = records_by_label[right_label]
            if left["benchmark"] != right["benchmark"]:
                continue
            left_tasks = {record.get("task_id") for record in left["records"]}
            right_tasks = {record.get("task_id") for record in right["records"]}
            if not left_tasks.intersection(right_tasks):
                continue
            rows.append(paired_comparison_row(left_label, right_label, left, right))
            if len(rows) >= max_pairs:
                return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(rows, columns=columns)


def jsonl_path_for_summary(root: Path, summary_path: Any) -> Path | None:
    if not isinstance(summary_path, str) or not summary_path:
        return None
    path = Path(summary_path)
    if not path.is_absolute() and not path.exists():
        path = root / path
    name = path.name
    if not name.endswith("_summary.json"):
        return None
    return path.with_name(name.replace("_summary.json", ".jsonl"))


def paired_comparison_row(
    left_label: str,
    right_label: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    from slm_steering.statistics.paired_tests import paired_bootstrap_test

    accuracy = paired_bootstrap_test(left["records"], right["records"], metric="success", n_bootstrap=200)
    tokens = paired_bootstrap_test(left["records"], right["records"], metric="tokens", n_bootstrap=200)
    return {
        "left_run": left_label,
        "right_run": right_label,
        "benchmark": left["benchmark"],
        "n_tasks": accuracy.n_tasks,
        "accuracy_delta": accuracy.mean_difference,
        "accuracy_ci_low": accuracy.ci_low,
        "accuracy_ci_high": accuracy.ci_high,
        "accuracy_p_value": accuracy.p_value_if_available,
        "token_delta": tokens.mean_difference,
        "token_ci_low": tokens.ci_low,
        "token_ci_high": tokens.ci_high,
        "dominance": dominance_label(accuracy.mean_difference, tokens.mean_difference),
    }


def dominance_label(accuracy_delta: float, token_delta: float) -> str:
    if pd.isna(accuracy_delta) or pd.isna(token_delta):
        return "insufficient_overlap"
    if accuracy_delta >= 0 and token_delta <= 0 and (accuracy_delta > 0 or token_delta < 0):
        return "right_dominates"
    if accuracy_delta <= 0 and token_delta >= 0 and (accuracy_delta < 0 or token_delta > 0):
        return "left_dominates"
    return "tradeoff"


def write_figures(frame: pd.DataFrame, figures_dir: Path, root: Path) -> list[Path]:
    paths = [
        write_accuracy_cost_plot(frame, figures_dir / "accuracy_cost_scatter.png"),
        write_phase_coverage_plot(frame, figures_dir / "phase_coverage.png"),
        write_probe_auc_plot(root, figures_dir / "probe_auc_by_layer.png"),
        write_steering_alpha_plot(frame, figures_dir / "steering_alpha_sweep.png"),
        write_adaptive_policy_plot(frame, figures_dir / "adaptive_budget_accuracy.png"),
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


def write_probe_auc_plot(root: Path, path: Path) -> Path:
    frames = []
    for csv_path in sorted((root / "runs" / "phase35").rglob("probe_results.csv")):
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        if {"layer", "auc"}.issubset(frame.columns):
            frame["source"] = csv_path.parent.name
            frames.append(frame)
    plt.figure(figsize=(7, 4.5))
    if frames:
        data = pd.concat(frames, ignore_index=True)
        data["layer"] = pd.to_numeric(data["layer"], errors="coerce")
        data["auc"] = pd.to_numeric(data["auc"], errors="coerce")
        data = data.dropna(subset=["layer", "auc"])
        if not data.empty:
            for source, group in data.groupby("source"):
                plt.plot(group["layer"], group["auc"], marker="o", label=source)
            plt.legend()
        else:
            plt.text(0.5, 0.5, "Sem resultados suficientes", ha="center", va="center")
    else:
        plt.text(0.5, 0.5, "Sem resultados suficientes", ha="center", va="center")
    plt.xlabel("Camada")
    plt.ylabel("Probe AUC")
    plt.title("Separabilidade latente por camada")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def write_steering_alpha_plot(frame: pd.DataFrame, path: Path) -> Path:
    plt.figure(figsize=(7, 4.5))
    required = {"phase", "alpha", "observed_best_of_n", "direction_type"}
    if required.issubset(frame.columns):
        data = frame[frame["phase"].eq("phase4")].copy()
        data["alpha"] = pd.to_numeric(data["alpha"], errors="coerce")
        data["observed_best_of_n"] = pd.to_numeric(data["observed_best_of_n"], errors="coerce")
        data = data.dropna(subset=["alpha", "observed_best_of_n"])
        if not data.empty:
            for label, group in data.groupby(["direction_type", "layer"], dropna=False):
                plt.plot(group["alpha"], group["observed_best_of_n"], marker="o", label=str(label))
            plt.legend(fontsize=8)
        else:
            plt.text(0.5, 0.5, "Sem resultados suficientes", ha="center", va="center")
    else:
        plt.text(0.5, 0.5, "Sem resultados suficientes", ha="center", va="center")
    plt.xlabel("Alpha")
    plt.ylabel("Acuracia Best-of-N observada")
    plt.title("Sweep causal de activation steering")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def write_adaptive_policy_plot(frame: pd.DataFrame, path: Path) -> Path:
    plt.figure(figsize=(7, 4.5))
    required = {"phase", "policy", "mean_tokens_until_success_or_budget", "observed_best_of_n"}
    if required.issubset(frame.columns):
        data = frame[frame["phase"].eq("phase5")].copy()
        data["mean_tokens_until_success_or_budget"] = pd.to_numeric(
            data["mean_tokens_until_success_or_budget"],
            errors="coerce",
        )
        data["observed_best_of_n"] = pd.to_numeric(data["observed_best_of_n"], errors="coerce")
        data = data.dropna(subset=["mean_tokens_until_success_or_budget", "observed_best_of_n"])
        if not data.empty:
            plt.scatter(data["mean_tokens_until_success_or_budget"], data["observed_best_of_n"], s=55)
            for _, row in data.iterrows():
                plt.annotate(str(row.get("policy") or row.get("run_label")), (row["mean_tokens_until_success_or_budget"], row["observed_best_of_n"]))
        else:
            plt.text(0.5, 0.5, "Sem resultados suficientes", ha="center", va="center")
    else:
        plt.text(0.5, 0.5, "Sem resultados suficientes", ha="center", va="center")
    plt.xlabel("Tokens ate sucesso ou orcamento")
    plt.ylabel("Acuracia")
    plt.title("Politicas adaptativas: custo vs acuracia")
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
