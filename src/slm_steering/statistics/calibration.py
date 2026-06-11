from __future__ import annotations

import numpy as np
import pandas as pd


def brier_score(probabilities, labels) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mask = np.isfinite(probs) & np.isfinite(y)
    if not mask.any():
        return float("nan")
    return float(np.mean((probs[mask] - y[mask]) ** 2))


def log_loss(probabilities, labels, *, eps: float = 1e-12) -> float:
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1 - eps)
    y = np.asarray(labels, dtype=np.float64)
    mask = np.isfinite(probs) & np.isfinite(y)
    if not mask.any():
        return float("nan")
    return float(-np.mean(y[mask] * np.log(probs[mask]) + (1 - y[mask]) * np.log(1 - probs[mask])))


def calibration_bins(probabilities, labels, *, n_bins: int = 10) -> pd.DataFrame:
    if n_bins < 1:
        raise ValueError("n_bins deve ser >= 1")
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mask = np.isfinite(probs) & np.isfinite(y)
    probs = probs[mask]
    y = y[mask]
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for bin_index in range(n_bins):
        low = edges[bin_index]
        high = edges[bin_index + 1]
        if bin_index == n_bins - 1:
            selected = (probs >= low) & (probs <= high)
        else:
            selected = (probs >= low) & (probs < high)
        count = int(selected.sum())
        rows.append(
            {
                "bin": bin_index,
                "probability_low": float(low),
                "probability_high": float(high),
                "count": count,
                "mean_predicted_probability": float(probs[selected].mean()) if count else np.nan,
                "empirical_success_rate": float(y[selected].mean()) if count else np.nan,
                "absolute_gap": (
                    abs(float(probs[selected].mean()) - float(y[selected].mean()))
                    if count
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(probabilities, labels, *, n_bins: int = 10) -> float:
    bins = calibration_bins(probabilities, labels, n_bins=n_bins)
    total = bins["count"].sum()
    if total == 0:
        return float("nan")
    weighted = bins["count"] * bins["absolute_gap"].fillna(0)
    return float(weighted.sum() / total)
