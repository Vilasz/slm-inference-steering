from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slm_steering.activations.latent_scores import compute_layerwise_latent_scores, summarize_latent_scores
from slm_steering.activations.probes import ProbeEvaluationConfig, evaluate_layerwise_probes
from slm_steering.activations.storage import load_activation_dataset
from slm_steering.reporting import read_jsonl
from slm_steering.statistics.bootstrap import (
    metric_best_of_k,
    metric_mean_attempts_until_success_or_budget,
    metric_mean_tokens,
    metric_pass_at_1,
    metric_resolution_rate,
    stratified_task_bootstrap,
)
from slm_steering.statistics.permutation import permutation_test_auc
from slm_steering.statistics.regression import build_success_regression_frame, compare_success_models
from slm_steering.statistics.reporting import write_json, write_table
from slm_steering.statistics.survival import (
    expected_attempts_to_success,
    hazard_by_attempt,
    kaplan_meier_curve,
    tokens_until_first_success,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fase 3.5: Statistical Latent Geometry.")
    parser.add_argument("--activations-dir", type=Path, default=Path("runs/phase3/qwen_phase3_probe"))
    parser.add_argument("--phase2-runs-dir", type=Path, default=Path("runs/phase2"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/phase35/latent_geometry"))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = analyze_latent_geometry(args)
    print(json.dumps(artifacts["summary"], indent=2, ensure_ascii=False))


def analyze_latent_geometry(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase2_runs = load_phase2_runs(args.phase2_runs_dir)
    if not (args.activations_dir / "activations.npz").exists():
        return write_empty_outputs(args, phase2_runs, reason=f"ativacoes ausentes em {args.activations_dir}")

    dataset = load_activation_dataset(args.activations_dir)
    manifest = dataset.manifest
    probe_results = evaluate_layerwise_probes(
        dataset,
        config=ProbeEvaluationConfig(train_fraction=args.train_fraction, seed=args.seed),
    )
    write_table(args.output_dir / "probe_results.csv", probe_results)

    latent_score_results = build_latent_score_results(dataset, seed=args.seed)
    write_table(args.output_dir / "latent_score_results.csv", latent_score_results)

    permutation_results = build_permutation_results(
        latent_score_results,
        probe_results,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    write_table(args.output_dir / "permutation_results.csv", permutation_results)

    regression_results = build_regression_results(latent_score_results, probe_results, seed=args.seed)
    write_table(args.output_dir / "regression_results.csv", regression_results)

    survival_results = build_survival_results(phase2_runs)
    write_table(args.output_dir / "survival_results.csv", survival_results)

    bootstrap_results = build_bootstrap_results(phase2_runs, n_bootstrap=args.n_bootstrap, seed=args.seed)
    write_table(args.output_dir / "bootstrap_results.csv", bootstrap_results)

    summary = build_summary(
        dataset=dataset,
        manifest=manifest,
        probe_results=probe_results,
        latent_score_results=latent_score_results,
        permutation_results=permutation_results,
        regression_results=regression_results,
        survival_results=survival_results,
        bootstrap_results=bootstrap_results,
        args=args,
    )
    write_json(args.output_dir / "summary.json", summary)
    return {"summary": summary}


def write_empty_outputs(
    args: argparse.Namespace,
    phase2_runs: dict[str, list[dict[str, Any]]],
    *,
    reason: str,
) -> dict[str, Any]:
    write_table(args.output_dir / "probe_results.csv", empty_probe_results())
    write_table(args.output_dir / "latent_score_results.csv", empty_latent_score_results())
    write_table(args.output_dir / "permutation_results.csv", empty_permutation_results())
    write_table(args.output_dir / "regression_results.csv", empty_regression_results())
    survival_results = build_survival_results(phase2_runs)
    bootstrap_results = build_bootstrap_results(phase2_runs, n_bootstrap=args.n_bootstrap, seed=args.seed)
    write_table(args.output_dir / "survival_results.csv", survival_results)
    write_table(args.output_dir / "bootstrap_results.csv", bootstrap_results)
    summary = {
        "phase": "phase35_statistical_latent_geometry",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "missing_activations",
        "reason": reason,
        "activations_dir": str(args.activations_dir),
        "phase2_runs_dir": str(args.phase2_runs_dir),
        "best_probe_layer": None,
        "best_probe_auc": None,
        "permutation_p_value": None,
        "latent_score_delta_auc": None,
        "best_survival_config": best_survival_config(survival_results),
        "mean_attempt_reduction_candidate": None,
        "recommended_steering_layers": [],
    }
    write_json(args.output_dir / "summary.json", summary)
    return {"summary": summary}


def build_latent_score_results(dataset, *, seed: int) -> pd.DataFrame:
    frames = []
    for direction_type in [
        "correctness_direction",
        "negative_correctness_direction",
        "random_direction",
        "shuffled_label_direction",
        "length_direction",
    ]:
        frame = compute_layerwise_latent_scores(dataset, direction_type=direction_type, seed=seed)
        if not frame.empty:
            if "model_id" not in frame.columns:
                frame["model_id"] = dataset.manifest.get("model_id", "unknown")
            frames.append(frame)
    if not frames:
        return empty_latent_score_results()
    return pd.concat(frames, ignore_index=True)


def build_permutation_results(
    latent_score_results: pd.DataFrame,
    probe_results: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    if latent_score_results.empty:
        return empty_permutation_results()
    candidate_layer = choose_best_layer(probe_results, latent_score_results)
    rows = []
    for direction_type, group in latent_score_results[latent_score_results["layer"].eq(candidate_layer)].groupby("direction_type"):
        if group["label"].nunique() < 2:
            continue
        groups = group["task_id"].to_numpy(dtype=object) if "task_id" in group.columns else None
        result = permutation_test_auc(
            group["latent_score"].to_numpy(dtype=float),
            group["label"].to_numpy(dtype=int),
            groups=groups,
            n_permutations=n_permutations,
            seed=seed,
        )
        rows.append(
            {
                "layer": candidate_layer,
                "direction_type": direction_type,
                **result.to_dict(),
            }
        )
    return pd.DataFrame(rows) if rows else empty_permutation_results()


def build_regression_results(
    latent_score_results: pd.DataFrame,
    probe_results: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    if latent_score_results.empty:
        return empty_regression_results()
    best_layer = choose_best_layer(probe_results, latent_score_results)
    frame = latent_score_results[
        latent_score_results["direction_type"].eq("correctness_direction")
        & latent_score_results["layer"].eq(best_layer)
    ].copy()
    if frame.empty:
        return empty_regression_results()
    regression_frame = build_success_regression_frame(frame)
    return compare_success_models(regression_frame, seed=seed)


def build_survival_results(phase2_runs: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for run_label, records in phase2_runs.items():
        curve = kaplan_meier_curve(records)
        for _, row in curve.iterrows():
            rows.append({"run_label": run_label, "analysis": "kaplan_meier", **row.to_dict()})
        hazard = hazard_by_attempt(records)
        for _, row in hazard.iterrows():
            rows.append({"run_label": run_label, "analysis": "hazard", **row.to_dict()})
        attempts = expected_attempts_to_success(records, censored="budget")
        tokens = tokens_until_first_success(records, censored="budget")
        rows.append(
            {
                "run_label": run_label,
                "analysis": "summary",
                "attempt": None,
                "mean_attempts": attempts.get("mean_attempts"),
                "median_attempts": attempts.get("median_attempts"),
                "mean_tokens": tokens.get("mean_tokens"),
                "median_tokens": tokens.get("median_tokens"),
                "n_tasks": attempts.get("n_tasks"),
                "n_events": attempts.get("n_events"),
            }
        )
    return pd.DataFrame(rows) if rows else empty_survival_results()


def build_bootstrap_results(
    phase2_runs: dict[str, list[dict[str, Any]]],
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for run_label, records in phase2_runs.items():
        if not records:
            continue
        max_k = max((len(record.get("attempts", [])) for record in records), default=1)
        metrics = {
            "pass_at_1": metric_pass_at_1,
            "resolution_rate": metric_resolution_rate,
            "mean_tokens": metric_mean_tokens,
            "mean_attempts_until_success_or_budget": metric_mean_attempts_until_success_or_budget,
            f"best_of_{max_k}": lambda sample, k=max_k: metric_best_of_k(sample, k=k),
        }
        for metric_name, metric_fn in metrics.items():
            result = stratified_task_bootstrap(
                records,
                metric_fn,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            rows.append(
                {
                    "run_label": run_label,
                    "metric": metric_name,
                    "metric_mean": result["metric_mean"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "n_bootstrap": result["n_bootstrap"],
                    "confidence_level": result["confidence_level"],
                }
            )
    return pd.DataFrame(rows) if rows else empty_bootstrap_results()


def build_summary(
    *,
    dataset,
    manifest: dict[str, Any],
    probe_results: pd.DataFrame,
    latent_score_results: pd.DataFrame,
    permutation_results: pd.DataFrame,
    regression_results: pd.DataFrame,
    survival_results: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    best_layer = choose_best_layer(probe_results, latent_score_results)
    best_probe_auc = best_probe_value(probe_results, "auc")
    correctness_perm = permutation_results[
        permutation_results.get("direction_type", pd.Series(dtype=str)).eq("correctness_direction")
    ] if not permutation_results.empty else pd.DataFrame()
    delta_auc = best_regression_delta(regression_results)
    return {
        "phase": "phase35_statistical_latent_geometry",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "activations_dir": str(args.activations_dir),
        "phase2_runs_dir": str(args.phase2_runs_dir),
        "model_id": manifest.get("model_id"),
        "num_activation_samples": dataset.num_samples,
        "num_layers": dataset.num_layers,
        "hidden_size": dataset.hidden_size,
        "num_tasks": int(dataset.metadata["task_id"].nunique()) if "task_id" in dataset.metadata.columns else None,
        "correct_samples": int(dataset.labels.sum()),
        "incorrect_samples": int((dataset.labels == 0).sum()),
        "best_probe_layer": best_layer,
        "best_probe_auc": best_probe_auc,
        "permutation_p_value": (
            float(correctness_perm["p_value"].iloc[0])
            if not correctness_perm.empty and "p_value" in correctness_perm
            else None
        ),
        "latent_score_delta_auc": delta_auc,
        "best_survival_config": best_survival_config(survival_results),
        "mean_attempt_reduction_candidate": mean_attempt_reduction_candidate(survival_results),
        "recommended_steering_layers": recommended_layers(probe_results, latent_score_results),
        "outputs": {
            "probe_results": str(args.output_dir / "probe_results.csv"),
            "latent_score_results": str(args.output_dir / "latent_score_results.csv"),
            "permutation_results": str(args.output_dir / "permutation_results.csv"),
            "regression_results": str(args.output_dir / "regression_results.csv"),
            "survival_results": str(args.output_dir / "survival_results.csv"),
            "bootstrap_results": str(args.output_dir / "bootstrap_results.csv"),
        },
        "caution": (
            "Analise piloto: interprete significancia e camadas candidatas com cautela "
            "quando houver poucas tarefas ou poucas tentativas corretas/incorretas."
        ),
    }


def load_phase2_runs(path: Path) -> dict[str, list[dict[str, Any]]]:
    runs = {}
    if not path.exists():
        return runs
    for jsonl_path in sorted(path.rglob("*.jsonl")):
        if "decisions" in jsonl_path.stem:
            continue
        try:
            records = read_jsonl(jsonl_path)
        except Exception:
            continue
        if records and isinstance(records[0], dict) and "attempts" in records[0]:
            runs[jsonl_path.stem] = records
    return runs


def choose_best_layer(probe_results: pd.DataFrame, latent_score_results: pd.DataFrame) -> int | None:
    if not probe_results.empty and "auc" in probe_results.columns:
        valid = probe_results.dropna(subset=["auc"])
        if not valid.empty:
            return int(valid.sort_values("auc", ascending=False).iloc[0]["layer"])
    if not latent_score_results.empty:
        summary = summarize_latent_scores(
            latent_score_results[latent_score_results["direction_type"].eq("correctness_direction")]
        )
        valid = summary.dropna(subset=["auc"])
        if not valid.empty:
            return int(valid.sort_values("auc", ascending=False).iloc[0]["layer"])
    return None


def best_probe_value(probe_results: pd.DataFrame, column: str) -> float | None:
    if probe_results.empty or column not in probe_results.columns:
        return None
    valid = probe_results[column].dropna()
    return float(valid.max()) if not valid.empty else None


def best_regression_delta(regression_results: pd.DataFrame) -> float | None:
    if regression_results.empty or "delta_auc_vs_observables" not in regression_results.columns:
        return None
    rows = regression_results[regression_results["model_name"].eq("observables_plus_latent")]
    if rows.empty:
        return None
    value = rows["delta_auc_vs_observables"].iloc[0]
    return float(value) if pd.notna(value) else None


def best_survival_config(survival_results: pd.DataFrame) -> str | None:
    if survival_results.empty or "analysis" not in survival_results.columns:
        return None
    summary = survival_results[survival_results["analysis"].eq("summary")].copy()
    if summary.empty or "mean_attempts" not in summary.columns:
        return None
    summary = summary.dropna(subset=["mean_attempts"])
    if summary.empty:
        return None
    return str(summary.sort_values("mean_attempts").iloc[0]["run_label"])


def mean_attempt_reduction_candidate(survival_results: pd.DataFrame) -> float | None:
    if survival_results.empty or "analysis" not in survival_results.columns:
        return None
    summary = survival_results[survival_results["analysis"].eq("summary")].dropna(subset=["mean_attempts"])
    if len(summary) < 2:
        return None
    return float(summary["mean_attempts"].max() - summary["mean_attempts"].min())


def recommended_layers(probe_results: pd.DataFrame, latent_score_results: pd.DataFrame, *, top_k: int = 3) -> list[int]:
    candidates = []
    if not probe_results.empty and "auc" in probe_results.columns:
        valid = probe_results.dropna(subset=["auc"]).sort_values("auc", ascending=False)
        candidates.extend([int(layer) for layer in valid["layer"].head(top_k).tolist()])
    if not latent_score_results.empty:
        summary = summarize_latent_scores(
            latent_score_results[latent_score_results["direction_type"].eq("correctness_direction")]
        )
        valid = summary.dropna(subset=["auc"]).sort_values("auc", ascending=False)
        candidates.extend([int(layer) for layer in valid["layer"].head(top_k).tolist()])
    unique = []
    for layer in candidates:
        if layer not in unique:
            unique.append(layer)
    return unique[:top_k]


def empty_probe_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "layer",
            "position",
            "probe_type",
            "auc",
            "accuracy",
            "f1",
            "brier_score",
            "calibration_error",
            "n_train_tasks",
            "n_test_tasks",
            "n_train_samples",
            "n_test_samples",
        ]
    )


def empty_latent_score_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample_index",
            "task_id",
            "attempt",
            "layer",
            "position",
            "direction_type",
            "latent_score",
            "label",
            "passed",
            "generated_tokens",
            "generation_seconds",
            "difficulty_class",
            "model_id",
        ]
    )


def empty_permutation_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["layer", "direction_type", "observed_score", "null_mean", "null_std", "p_value", "n_permutations"]
    )


def empty_regression_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model_name",
            "auc",
            "brier_score",
            "log_loss",
            "accuracy",
            "n_train_tasks",
            "n_test_tasks",
            "n_train_samples",
            "n_test_samples",
            "coefficients_json",
            "odds_ratios_json",
            "delta_auc_vs_observables",
        ]
    )


def empty_survival_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "run_label",
            "analysis",
            "attempt",
            "at_risk",
            "events",
            "censored",
            "survival_probability",
            "cumulative_resolution",
            "hazard",
            "mean_attempts",
            "mean_tokens",
            "n_tasks",
            "n_events",
        ]
    )


def empty_bootstrap_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["run_label", "metric", "metric_mean", "ci_low", "ci_high", "n_bootstrap", "confidence_level"]
    )


if __name__ == "__main__":
    main()
