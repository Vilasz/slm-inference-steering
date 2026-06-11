from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_survival_table(records: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return _survival_from_attempt_frame(records)
    rows = []
    for record in records:
        attempts = record.get("attempts", [])
        first_success = record.get("first_success_attempt")
        event_observed = first_success is not None
        time = int(first_success if event_observed else len(attempts))
        rows.append(
            {
                "task_id": record.get("task_id"),
                "time": time,
                "event_observed": bool(event_observed),
                "first_success_attempt": first_success,
                "max_attempts": len(attempts),
                "tokens_until_event_or_censor": _tokens_until(attempts, time),
                "seconds_until_event_or_censor": _seconds_until(attempts, time),
                "total_tokens": _tokens_until(attempts, len(attempts)),
                "total_seconds": _seconds_until(attempts, len(attempts)),
            }
        )
    return pd.DataFrame(rows)


def kaplan_meier_curve(records: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    table = build_survival_table(records)
    if table.empty:
        return pd.DataFrame(
            columns=["attempt", "at_risk", "events", "censored", "survival_probability", "cumulative_resolution"]
        )
    max_time = int(table["time"].max())
    survival = 1.0
    rows = []
    for attempt in range(1, max_time + 1):
        at_risk = int((table["time"] >= attempt).sum())
        events = int(((table["time"] == attempt) & table["event_observed"]).sum())
        censored = int(((table["time"] == attempt) & ~table["event_observed"]).sum())
        if at_risk > 0:
            survival *= 1 - events / at_risk
        rows.append(
            {
                "attempt": attempt,
                "at_risk": at_risk,
                "events": events,
                "censored": censored,
                "survival_probability": float(survival),
                "cumulative_resolution": float(1 - survival),
            }
        )
    return pd.DataFrame(rows)


def hazard_by_attempt(records: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    curve = kaplan_meier_curve(records)
    if curve.empty:
        return pd.DataFrame(columns=["attempt", "at_risk", "events", "hazard"])
    frame = curve[["attempt", "at_risk", "events"]].copy()
    frame["hazard"] = frame.apply(
        lambda row: float(row["events"] / row["at_risk"]) if row["at_risk"] else np.nan,
        axis=1,
    )
    return frame


def expected_attempts_to_success(
    records: list[dict[str, Any]] | pd.DataFrame,
    *,
    censored: str = "budget",
) -> dict[str, float | int | str | None]:
    table = build_survival_table(records)
    if table.empty:
        return {"mean_attempts": None, "n_tasks": 0, "censored": censored}
    if censored == "ignore":
        values = table.loc[table["event_observed"], "time"].to_numpy(dtype=np.float64)
    elif censored == "budget":
        values = table["time"].to_numpy(dtype=np.float64)
    else:
        raise ValueError("censored deve ser 'budget' ou 'ignore'")
    return {
        "mean_attempts": float(values.mean()) if len(values) else None,
        "median_attempts": float(np.median(values)) if len(values) else None,
        "n_tasks": int(len(table)),
        "n_events": int(table["event_observed"].sum()),
        "censored": censored,
    }


def tokens_until_first_success(
    records: list[dict[str, Any]] | pd.DataFrame,
    *,
    censored: str = "budget",
) -> dict[str, float | int | str | None]:
    table = build_survival_table(records)
    if table.empty:
        return {"mean_tokens": None, "n_tasks": 0, "censored": censored}
    if censored == "ignore":
        values = table.loc[table["event_observed"], "tokens_until_event_or_censor"].to_numpy(dtype=np.float64)
    elif censored == "budget":
        values = table["tokens_until_event_or_censor"].to_numpy(dtype=np.float64)
    else:
        raise ValueError("censored deve ser 'budget' ou 'ignore'")
    return {
        "mean_tokens": float(values.mean()) if len(values) else None,
        "median_tokens": float(np.median(values)) if len(values) else None,
        "n_tasks": int(len(table)),
        "n_events": int(table["event_observed"].sum()),
        "censored": censored,
    }


def _survival_from_attempt_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"task_id", "attempt", "passed"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Colunas ausentes: {sorted(missing)}")
    rows = []
    for task_id, group in frame.sort_values("attempt").groupby("task_id", sort=False):
        passed = group[group["passed"].astype(bool)]
        first_success = int(passed["attempt"].iloc[0]) if not passed.empty else None
        event = first_success is not None
        time = first_success if event else int(group["attempt"].max())
        rows.append(
            {
                "task_id": task_id,
                "time": int(time),
                "event_observed": bool(event),
                "first_success_attempt": first_success,
                "max_attempts": int(group["attempt"].max()),
                "tokens_until_event_or_censor": float(
                    group[group["attempt"] <= time].get("generated_tokens", pd.Series(dtype=float)).fillna(0).sum()
                ),
                "seconds_until_event_or_censor": float(
                    group[group["attempt"] <= time].get("generation_seconds", pd.Series(dtype=float)).fillna(0).sum()
                ),
                "total_tokens": float(group.get("generated_tokens", pd.Series(dtype=float)).fillna(0).sum()),
                "total_seconds": float(group.get("generation_seconds", pd.Series(dtype=float)).fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _tokens_until(attempts: list[dict[str, Any]], cutoff: int) -> float:
    return float(sum(float(attempt.get("generated_tokens") or 0) for attempt in attempts[:cutoff]))


def _seconds_until(attempts: list[dict[str, Any]], cutoff: int) -> float:
    return float(
        sum(
            float(attempt.get("generation_seconds") or 0)
            + float(attempt.get("verification_seconds") or 0)
            for attempt in attempts[:cutoff]
        )
    )
