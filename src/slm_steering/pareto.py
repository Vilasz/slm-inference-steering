from __future__ import annotations

from typing import Any

import pandas as pd

from slm_steering.analysis import RunBundle


def pareto_frame(runs: list[RunBundle]) -> pd.DataFrame:
    rows = []
    for run in runs:
        summary = run.summary
        config = summary.get("config", {})
        matrix = summary.get("matrix", {})
        phase2 = summary.get("phase2", {})
        num_tasks = summary.get("num_tasks") or 0
        total_tokens = summary.get("total_generated_tokens") or 0
        total_seconds = (
            (summary.get("total_generation_seconds") or 0)
            + (summary.get("total_verification_seconds") or 0)
        )
        rows.append(
            {
                "run": run.label,
                "model_id": phase2.get("model_id") or matrix.get("model_id") or config.get("model_id"),
                "model_key": phase2.get("model_key") or matrix.get("model_key"),
                "benchmark": phase2.get("benchmark") or config.get("benchmark") or config.get("dataset"),
                "temperature": _first_present(phase2, matrix, config, key="temperature"),
                "top_p": _first_present(phase2, matrix, config, key="top_p"),
                "n": phase2.get("n") or matrix.get("n") or config.get("n") or summary.get("requested_n"),
                "early_stop": phase2.get("early_stop") if "early_stop" in phase2 else config.get("early_stop"),
                "accuracy": summary.get("observed_best_of_n"),
                "pass_at_1": summary.get("strict_pass_at_1"),
                "mean_generated_tokens": total_tokens / num_tasks if num_tasks else None,
                "mean_effective_tokens": summary.get("mean_tokens_until_success_or_budget"),
                "mean_latency_seconds": total_seconds / num_tasks if num_tasks else None,
                "mean_effective_latency_seconds": summary.get("mean_seconds_until_success_or_budget"),
                "mean_attempts_until_success_or_budget": summary.get("mean_attempts_until_success_or_budget"),
                "total_attempts": summary.get("total_attempts"),
                "solved_tasks": summary.get("solved_tasks"),
                "num_tasks": num_tasks,
                "tokens_per_solved_task": summary.get("generated_tokens_per_solved_task"),
                "effective_tokens_per_solved_task": summary.get("effective_tokens_per_solved_task"),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["is_pareto_efficient"] = pareto_efficient_mask(
            frame,
            cost_col="mean_effective_tokens",
            quality_col="accuracy",
        )
    return frame


def pareto_efficient_mask(
    frame: pd.DataFrame,
    *,
    cost_col: str,
    quality_col: str,
) -> list[bool]:
    mask = []
    for idx, row in frame.iterrows():
        cost = row[cost_col]
        quality = row[quality_col]
        if pd.isna(cost) or pd.isna(quality):
            mask.append(False)
            continue
        dominated = False
        for other_idx, other in frame.iterrows():
            if other_idx == idx:
                continue
            other_cost = other[cost_col]
            other_quality = other[quality_col]
            if pd.isna(other_cost) or pd.isna(other_quality):
                continue
            no_worse = other_cost <= cost and other_quality >= quality
            strictly_better = other_cost < cost or other_quality > quality
            if no_worse and strictly_better:
                dominated = True
                break
        mask.append(not dominated)
    return mask


def _first_present(*mappings: dict[str, Any], key: str) -> Any:
    for mapping in mappings:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None
