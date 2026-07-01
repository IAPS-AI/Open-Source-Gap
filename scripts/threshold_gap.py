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


def _gaussian_smooth_grid(dates_int: np.ndarray, values: np.ndarray,
                          grid_int: np.ndarray, bandwidth: float,
                          min_ess: float = SMOOTH_MIN_ESS) -> np.ndarray:
    """Gaussian-kernel weighted mean at each grid point (vectorised).
    NaN where effective sample size sum(w) < min_ess."""
    dt = grid_int[:, None] - dates_int[None, :]  # (g, n)
    w = np.exp(-0.5 * (dt / bandwidth) ** 2)
    sw = w.sum(axis=1)
    num = (w * values[None, :]).sum(axis=1)
    out = np.full(sw.shape, np.nan, dtype=float)
    ok = sw >= min_ess
    out[ok] = num[ok] / sw[ok]
    return out


def gaussian_smooth_with_ci(
    dates: pd.Series,
    values: pd.Series,
    *,
    bandwidth_days: float = SMOOTH_BANDWIDTH_DAYS,
    step_days: int = SMOOTH_STEP_DAYS,
    n_boot: int = SMOOTH_N_BOOT,
    ci: float = SMOOTH_CI,
    min_ess: float = SMOOTH_MIN_ESS,
    seed: int = SMOOTH_SEED,
):
    """Nadaraya-Watson Gaussian-kernel smoother with bootstrap CI on a
    uniform date grid. Returns (grid_dates, mean, lo, hi) as pd.Series."""
    dates = pd.to_datetime(pd.Series(list(dates))).reset_index(drop=True)
    if dates.empty:
        empty_d = pd.Series([], dtype="datetime64[ns]")
        empty_f = pd.Series([], dtype=float)
        return empty_d, empty_f, empty_f.copy(), empty_f.copy()
    values = pd.Series(list(values)).reset_index(drop=True).astype(float).values

    d_int = dates.values.astype("datetime64[D]").astype(np.int64)
    grid = pd.date_range(dates.min(), dates.max(), freq=f"{step_days}D")
    grid_int = grid.values.astype("datetime64[D]").astype(np.int64)
    h = float(bandwidth_days)

    central = _gaussian_smooth_grid(d_int, values, grid_int, h, min_ess)

    n = len(values)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(grid_int)), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = _gaussian_smooth_grid(d_int[idx], values[idx], grid_int, h, min_ess)

    alpha = (1 - ci) / 2
    import warnings
    with warnings.catch_warnings():
        # All-NaN grid columns (outside every bootstrap's support) are expected.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        lo = np.nanquantile(boot, alpha, axis=0)
        hi = np.nanquantile(boot, 1 - alpha, axis=0)
    return pd.Series(grid), pd.Series(central), pd.Series(lo), pd.Series(hi)


def _iso(ts) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat()


def summarize_datapoints(datapoints: list[dict]) -> dict:
    accepted = [r for r in datapoints if r["accepted"]]
    med = (round(float(np.median([r["gap_months"] for r in accepted])), 1)
           if accepted else None)
    return {
        "n_thresholds": len(datapoints),
        "n_valid": sum(1 for r in datapoints if r["valid"]),
        "n_accepted": len(accepted),
        "n_still_open": sum(1 for r in datapoints if r["still_open"]),
        "median_gap_months": med,
    }


def build_threshold_analysis(
    df: pd.DataFrame,
    benchmark_id: str,
    *,
    score_col: str = "score",
    model_col: str = "model",
    date_col: str = "date",
    open_col: str = "Open",
    as_of=None,
) -> dict:
    """JSON-ready `threshold_analysis` block for one benchmark."""
    review = THRESHOLD_REVIEW.get(benchmark_id, {})
    datapoints = compute_threshold_datapoints(
        df,
        score_col=score_col,
        model_col=model_col,
        date_col=date_col,
        open_col=open_col,
        thresholds=review.get("thresholds"),
        accepted_thresholds=review.get("accepted_thresholds"),
        validity_floor=review.get("validity_floor"),
        as_of=as_of,
    )
    out_points = [
        {
            **r,
            "first_closed_date": _iso(r["first_closed_date"]),
            "first_open_date": _iso(r["first_open_date"]),
        }
        for r in datapoints
    ]
    floor = (out_points[0]["validity_floor"] if out_points
             else review.get("validity_floor"))
    accepted = review.get("accepted_thresholds")
    config = {
        "keep": bool(review.get("keep", False)),
        "data_access": review.get("data_access"),
        "validity_floor": floor,
        "thresholds": ([float(x) for x in review["thresholds"]]
                       if review.get("thresholds") else None),
        "accepted_thresholds": ([float(x) for x in accepted]
                                if accepted else None),
        "source_review": review.get("source_review"),
    }
    return {
        "config": config,
        "datapoints": out_points,
        "summary": summarize_datapoints(datapoints),
    }


def build_threshold_aggregate(benchmarks: dict) -> dict:
    """Pool accepted datapoints from keep=True benchmarks; smooth the gap
    over time (overall + public/private split), mirroring the source's
    combined-by-access analysis. `benchmarks` maps benchmark id -> processed
    benchmark dict (with `metadata` and `threshold_analysis`)."""
    datapoints: list[dict] = []
    for bid, bench in benchmarks.items():
        ta = (bench or {}).get("threshold_analysis")
        if not ta or not ta.get("config", {}).get("keep"):
            continue
        access = ta["config"].get("data_access")
        name = (bench.get("metadata") or {}).get("name", bid)
        for r in ta["datapoints"]:
            if not r.get("accepted"):
                continue
            datapoints.append({
                "benchmark_id": bid,
                "benchmark_name": name,
                "data_access": access,
                "threshold": r["threshold"],
                "first_closed_model": r["first_closed_model"],
                "first_closed_date": r["first_closed_date"],
                "first_open_model": r["first_open_model"],
                "first_open_date": r["first_open_date"],
                "gap_days": r["gap_days"],
                "gap_months": r["gap_months"],
            })
    datapoints.sort(key=lambda p: (p["first_open_date"], p["benchmark_id"]))

    def trend_for(points: list[dict]):
        if len(points) < 2:
            return None
        grid, mean, lo, hi = gaussian_smooth_with_ci(
            pd.Series([p["first_open_date"] for p in points]),
            pd.Series([p["gap_months"] for p in points]),
        )
        out = []
        for g, m, l, h in zip(grid, mean, lo, hi):
            if not np.isfinite(m):
                continue
            out.append({
                "date": pd.Timestamp(g).date().isoformat(),
                "mean": round(float(m), 2),
                "lo": round(float(l), 2) if np.isfinite(l) else None,
                "hi": round(float(h), 2) if np.isfinite(h) else None,
            })
        if not out:
            return None
        return {
            "points": out,
            "n_datapoints": len(points),
            "n_benchmarks": len({p["benchmark_id"] for p in points}),
        }

    groups = {
        "overall": datapoints,
        "public": [p for p in datapoints if p["data_access"] == "public"],
        "private": [p for p in datapoints if p["data_access"] == "private"],
    }
    trends: dict = {}
    medians: dict = {}
    for key, pts in groups.items():
        t = trend_for(pts)
        if t is not None:
            trends[key] = t
        if pts:
            medians[key] = round(
                float(np.median([p["gap_months"] for p in pts])), 1)

    return {
        "datapoints": datapoints,
        "trends": trends,
        "medians": medians,
        "parameters": {
            "bandwidth_days": SMOOTH_BANDWIDTH_DAYS,
            "step_days": SMOOTH_STEP_DAYS,
            "n_boot": SMOOTH_N_BOOT,
            "ci": SMOOTH_CI,
            "min_ess": SMOOTH_MIN_ESS,
            "seed": SMOOTH_SEED,
        },
        "source": "open_closed_gap threshold-crossing methodology",
    }
