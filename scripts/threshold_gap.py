"""Threshold-crossing gap analysis, ported from Havard Ihle's open_closed_gap
(https://github.com/htihle/open_closed_gap).

For each score threshold T on a fixed grid: gap = release date of the first
open-weight model to score >= T minus release date of the first closed model
to do so. Where the source repo's logic conflicts with this repo's (open/
closed classification, live data, percentage score scale, month units), this
repo's logic prevails. See
docs/specs/2026-07-01-threshold-gap-integration-design.md.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 365.25 / 12  # matches scripts/update_data.py

# Gaussian-smoother parameters — the source repo's published-figure settings.
SMOOTH_BANDWIDTH_DAYS = 60.0
SMOOTH_STEP_DAYS = 7
SMOOTH_N_BOOT = 5000
SMOOTH_CI = 0.90
SMOOTH_MIN_ESS = 2.0
SMOOTH_SEED = 0

# Per-benchmark verdicts from the source repo's manual per-pair review,
# mapped to this repo's benchmark ids. Score thresholds are on THIS repo's
# scale (0-1 benchmarks stored as percentages, so source values x100; METR
# horizons are minutes in both). keep=False benchmarks still get per-benchmark
# datapoints but are excluded from the cross-benchmark aggregate (source
# DISCARD verdicts; ECI excluded as a composite index).
THRESHOLD_REVIEW: dict[str, dict] = {
    "gpqa_diamond": {
        "accepted_thresholds": [35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0,
                                70.0, 75.0, 80.0, 85.0, 90.0],
        "validity_floor": 20.0,
        "data_access": "public",
        "keep": True,
        "source_review": "open_closed_gap",
    },
    "math_level_5": {
        "accepted_thresholds": [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0,
                                50.0, 55.0, 60.0, 65.0, 80.0, 90.0, 95.0],
        "validity_floor": 10.0,
        "data_access": "public",
        "keep": True,
        "source_review": "open_closed_gap",
    },
    "otis_mock_aime": {
        "accepted_thresholds": [5.0, 10.0, 15.0, 20.0, 45.0, 50.0, 55.0,
                                60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0,
                                95.0],
        "validity_floor": 5.0,
        "data_access": "public",
        "keep": True,
        "source_review": "open_closed_gap",
    },
    "frontiermath_public": {
        "accepted_thresholds": [10.0, 15.0, 25.0, 30.0, 35.0],
        "validity_floor": 10.0,
        "data_access": "private",
        "keep": True,
        "source_review": "open_closed_gap",
    },
    "metr_time_horizon": {
        "thresholds": [2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
        "accepted_thresholds": [2.0, 5.0, 10.0, 20.0, 50.0],
        "validity_floor": 1.5,
        "data_access": "private",
        "keep": True,
        "source_review": "open_closed_gap",
    },
    # Source verdict: DISCARD (excluded from its aggregate).
    "swe_bench_verified": {
        "data_access": "public",
        "keep": False,
        "source_review": "open_closed_gap",
    },
    # Source verdict: reviewed, 0 datapoints accepted.
    "simpleqa_verified": {
        "validity_floor": 20.0,
        "data_access": "public",
        "keep": False,
        "source_review": "open_closed_gap",
    },
    # Source verdict: DISCARD.
    "chess_puzzles": {
        "data_access": "private",
        "keep": False,
        "source_review": "open_closed_gap",
    },
    # Not a source benchmark: composite index, would double-count its
    # constituent benchmarks in the aggregate.
    "eci": {
        "data_access": None,
        "keep": False,
        "source_review": None,
    },
}


def auto_thresholds(scores: pd.Series) -> np.ndarray:
    s_max = float(scores.max())
    if s_max <= 1.05:
        return np.arange(0.05, 1.0, 0.05)
    if s_max <= 110:
        return np.arange(5.0, 100.0, 5.0)
    s_min = float(max(scores.min(), 0))
    return np.linspace(s_min, s_max, 21)[1:-1]


def auto_validity_floor(scores: pd.Series) -> float:
    """Below the 10th percentile of observed scores, "first crosser" mostly
    reflects test order rather than capability (source heuristic)."""
    return float(np.percentile(scores, 10))


def first_crosser(df: pd.DataFrame, threshold: float,
                  score_col: str = "score", date_col: str = "date"):
    """Earliest-dated row with score >= threshold, or None."""
    hits = df[df[score_col] >= threshold]
    if hits.empty:
        return None
    return hits.sort_values(date_col, kind="mergesort").iloc[0]


def compute_threshold_datapoints(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    model_col: str = "model",
    date_col: str = "date",
    open_col: str = "Open",
    thresholds=None,
    accepted_thresholds=None,
    validity_floor: float | None = None,
    as_of=None,
) -> list[dict]:
    """One dict per grid threshold a closed model has reached.

    Expects `open_col` to hold this repo's open/closed classification.
    Dates in the returned dicts are pandas Timestamps (callers serialise).
    """
    d = df.dropna(subset=[date_col, score_col]).copy()
    if d.empty or open_col not in d.columns:
        return []
    is_open = d[open_col].astype(bool)
    closed = d[~is_open]
    open_df = d[is_open]
    if closed.empty or open_df.empty:
        return []

    if thresholds is None:
        thresholds = auto_thresholds(d[score_col])
    thresholds = np.asarray(list(thresholds), dtype=float)
    if thresholds.size == 0:
        return []

    floor = validity_floor
    if floor is None:
        floor = auto_validity_floor(d[score_col])

    allow = None
    if accepted_thresholds is not None:
        allow = {round(float(x), 6) for x in accepted_thresholds}

    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(datetime.now())

    rows: list[dict] = []
    for t in thresholds:
        c = first_crosser(closed, float(t), score_col, date_col)
        if c is None:
            continue  # not reached by closed side either
        o = first_crosser(open_df, float(t), score_col, date_col)
        if o is None:
            # Closed reached T; open pending — ongoing gap vs as_of.
            gap_so_far_days = (as_of_ts - c[date_col]).days
            rows.append({
                "threshold": float(t),
                "first_closed_model": str(c[model_col]),
                "first_closed_date": c[date_col],
                "first_open_model": None,
                "first_open_date": None,
                "gap_days": None,
                "gap_months": None,
                "gap_months_so_far": round(gap_so_far_days / DAYS_PER_MONTH, 1),
                "valid": False,
                "accepted": False,
                "still_open": True,
                "validity_floor": float(floor),
            })
            continue
        gap_days = int((o[date_col] - c[date_col]).days)
        if allow is not None:
            is_valid = round(float(t), 6) in allow
        else:
            is_valid = float(t) > floor
        rows.append({
            "threshold": float(t),
            "first_closed_model": str(c[model_col]),
            "first_closed_date": c[date_col],
            "first_open_model": str(o[model_col]),
            "first_open_date": o[date_col],
            "gap_days": gap_days,
            "gap_months": round(gap_days / DAYS_PER_MONTH, 1),
            "gap_months_so_far": None,
            "valid": bool(is_valid),
            "accepted": False,
            "still_open": False,
            "validity_floor": float(floor),
        })

    # The same (closed, open) pair recurs across a contiguous run of
    # thresholds with an identical gap. Keep ONE representative datapoint per
    # accepted pair, at the HIGHEST threshold the pair spans (even when that
    # exact threshold is not in the allowlist — the source expands accepted
    # anchor pairs to the pair's full span).
    rows.sort(key=lambda r: r["threshold"])
    accepted_keys = {
        (r["first_closed_model"], r["first_open_model"])
        for r in rows
        if r["valid"]
    }
    for key in accepted_keys:
        members = [
            r for r in rows
            if (r["first_closed_model"], r["first_open_model"]) == key
        ]
        members[-1]["accepted"] = True
    return rows
