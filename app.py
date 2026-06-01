"""
ECI Accessibility Gap Web Application

An interactive visualization of the performance gap between open and closed-source
AI models on the Epoch AI ECI (Effective Compute Index).
"""

import csv
import io
import logging
import math
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, send_from_directory
from scipy.stats import linregress, norm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DAYS_PER_MONTH = 365.25 / 12  # 30.4375 - accurate average days per month
ECI_MATCH_THRESHOLD = 1.0  # ECI points - "approximately equal" when no CI is available
# One-sided 5% critical value: P(Z <= 1.6448) = 0.95. See _open_caught_up.
Z_ONE_SIDED_05 = float(norm.ppf(0.95))  # 1.6448...

app = Flask(__name__)

# Data source
# benchmarked_models.csv: one row per (Model, variant) with Organization,
# Version release date, Model accessibility, and ECI inline. We aggregate
# below to one row per Model (best variant by ECI).
BENCHMARKED_MODELS_URL = "https://epoch.ai/data/benchmarked_models.csv"

# Cache duration (5 minutes)
CACHE_DURATION = timedelta(minutes=5)
_cache: dict[str, Any] = {"data": None, "timestamp": None}


def get_rank(
    df: pd.DataFrame,
    n: int | None = None,
    sort_col: str = "date",
    val_col: str = "eci",
) -> pd.Series:
    """
    Cumulative rank of *val_col* up to each row, ordered by *sort_col*,
    robust to missing values.

    • If *val_col* is NaN for a row → rank is NaN.
    • Rows whose *val_col* is NaN do **not** affect later ranks.
    • Rows whose *sort_col* is NaN are treated as having unknown release time
      → their own rank is NaN and they do not affect others.
    • If *n* is given, ranks > n are set to NaN (frontier filter).

    Returns
    -------
    pd.Series aligned with *df.index* (dtype float, so NaNs are allowed).
    """
    ordered = df.sort_values(sort_col, kind="mergesort", na_position="last").reset_index()
    vals = ordered[val_col]
    ranks = pd.Series(np.nan, index=ordered.index, dtype=float)

    seen = []
    for idx, v in enumerate(vals):
        if pd.isna(v):
            continue
        rank = 1 + sum(prev > v for prev in seen)
        ranks.iloc[idx] = rank
        seen.append(v)

    if n is not None:
        ranks = ranks.where(ranks <= n)

    ranks.index = ordered["index"]
    return ranks.reindex(df.index)


def check_statistical_diff(row_open: pd.Series, row_closed: pd.Series, alpha: float = 0.05) -> bool:
    """
    H0: eci_open == eci_closed
    Two-sided z-test using combined SE = sqrt(s1^2 + s2^2).
    Returns True if |diff| is significant at level alpha, else False.
    Uses eci_std derived from confidence intervals.
    """
    m1 = row_open.get("eci")
    m2 = row_closed.get("eci")
    s1 = row_open.get("eci_std")
    s2 = row_closed.get("eci_std")

    if any(pd.isna(x) for x in (m1, m2, s1, s2)):
        # If uncertainty data is missing, fall back to exact comparison
        return True if pd.notna(m1) and pd.notna(m2) and m1 != m2 else False

    se = math.sqrt(s1**2 + s2**2)
    if se == 0:
        return abs(m1 - m2) > 0

    z = abs(m1 - m2) / se
    zcrit = float(norm.ppf(1 - alpha / 2))

    return z > zcrit
    # Note: This function is currently unused in the strict matching logic,
    # but kept for potential future use or reference.


def fetch_eci_data() -> pd.DataFrame:
    """Fetch ECI scores from Epoch AI with caching."""
    now = datetime.now()

    if _cache["data"] is not None and _cache["timestamp"] is not None:
        if now - _cache["timestamp"] < CACHE_DURATION:
            logger.info("Serving data from cache")
            return _cache["data"].copy()

    logger.info("Fetching data from Epoch AI (benchmarked_models.csv)...")
    try:
        # Parse via stdlib csv: pandas mistokenizes quoted multi-line
        # Description fields in benchmarked_models.csv.
        resp = requests.get(BENCHMARKED_MODELS_URL, timeout=30)
        resp.raise_for_status()
        df = pd.DataFrame(list(csv.DictReader(io.StringIO(resp.text))))

        # benchmarked_models.csv ships one row per (Model, variant). Keep
        # the best variant per Model and rename Version release date → date.
        # csv.DictReader returns strings; coerce numeric columns.
        for col in ("eci", "eci_ci_low", "eci_ci_high"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "Version release date" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"Version release date": "date"})
        df = df.dropna(subset=["eci"])
        df = df.sort_values("eci", ascending=False, kind="mergesort")
        df = df.drop_duplicates(subset=["Model"], keep="first").reset_index(drop=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
        # Derive eci_std from confidence intervals (assuming 90% CI)
        # For 90% CI, z = 1.645, so std = (ci_high - ci_low) / (2 * 1.645)
        if "eci_ci_low" in df.columns and "eci_ci_high" in df.columns:
            df["eci_std"] = (df["eci_ci_high"] - df["eci_ci_low"]) / (2 * 1.645)
        else:
            df["eci_std"] = pd.NA
        
        _cache["data"] = df
        _cache["timestamp"] = now
        return df.copy()
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        if _cache["data"] is not None:
            return _cache["data"].copy()
        raise


def process_data() -> dict[str, Any]:
    """Process ECI data and calculate gaps."""
    df = fetch_eci_data()

    # Classify models as open or closed: any "Open weights (...)" variant
    # is open; everything else (API access, Hosted access, Unreleased,
    # missing) is closed.
    df["Open"] = df["Model accessibility"].str.contains("Open", na=False)

    # Get rankings for each group
    df_open = df[df["Open"]].copy()
    df_closed = df[~df["Open"]].copy()

    df_open["group_rank"] = get_rank(df_open, sort_col="date", val_col="eci")
    df_closed["group_rank"] = get_rank(df_closed, sort_col="date", val_col="eci")

    # Combine and filter to frontier models (rank 1)
    df_combined = pd.concat([df_open, df_closed]).sort_values("date")
    df_frontier = df_combined[df_combined["group_rank"] <= 1].copy()

    # Prepare model data for visualization (Frontier only)
    models = []
    for _, row in df_frontier.iterrows():
        models.append({
            "model": row.get("Model", row.get("model version", "Unknown")),
            "display_name": row.get("Model", row.get("Model", "Unknown")),
            "eci": float(row["eci"]) if pd.notna(row["eci"]) else None,
            "eci_std": float(row["eci_std"]) if pd.notna(row.get("eci_std")) else None,
            "date": row["date"].isoformat() if pd.notna(row["date"]) else None,
            "organization": row.get("Organization", "Unknown"),
            "is_open": bool(row["Open"]),
        })

    # Prepare ALL models for trend visualization
    trend_models = []
    # Filter df_combined to ensure we have valid ECI and dates
    df_all_valid = df_combined.dropna(subset=["eci", "date"])
    for _, row in df_all_valid.iterrows():
        trend_models.append({
            "model": row.get("Model", row.get("model version", "Unknown")),
            "display_name": row.get("Display name", row.get("Model", "Unknown")),
            "eci": float(row["eci"]),
            "eci_std": float(row["eci_std"]) if pd.notna(row.get("eci_std")) else None,
            "date": row["date"].isoformat(),
            "organization": row.get("Organization", "Unknown"),
            "is_open": bool(row["Open"]),
        })

    # Calculate horizontal gaps (time for open to catch up) - still frontier based
    gaps = calculate_horizontal_gaps(df_frontier)

    # Calculate statistics - still frontier based
    stats = calculate_statistics(df_frontier, gaps)

    # Calculate trends - USING ALL MODELS NOW
    trends = calculate_trends(df_all_valid)

    return {
        "models": models,
        "trend_models": trend_models,
        "gaps": gaps,
        "statistics": stats,
        "trends": trends,
        "last_updated": datetime.now().isoformat(),
    }


def calculate_horizontal_gaps(df: pd.DataFrame) -> list[dict]:
    """
    Calculate horizontal gaps: for each closed model, find the first open model
    released AFTER the closed model that matches or exceeds its ECI score.
    """
    df_open = df[df["Open"]].sort_values("date")
    df_closed = df[~df["Open"]].sort_values("date")

    gaps = []

    for _, closed_row in df_closed.iterrows():
        closed_eci = closed_row["eci"]
        closed_date = closed_row["date"]

        if pd.isna(closed_eci) or pd.isna(closed_date):
            continue

        # Find first open model released AFTER this closed model that matches or exceeds this ECI
        matching_open = None
        match_type = None  # 'exact' or 'statistical'
        for _, open_row in df_open.iterrows():
            if pd.isna(open_row["eci"]) or pd.isna(open_row["date"]):
                continue

            # Only consider open models released after the closed model
            if open_row["date"] <= closed_date:
                continue

            # Check if open model is >= closed model's ECI (with 1 point tolerance)
            if open_row["eci"] >= closed_eci - 1:
                matching_open = open_row
                match_type = "exact"
                break

        if matching_open is not None:
            gap_days = (matching_open["date"] - closed_date).days
            gap_months = gap_days / DAYS_PER_MONTH

            gaps.append({
                "closed_model": closed_row.get("Model", "Unknown"),
                "closed_date": closed_date.isoformat(),
                "closed_eci": float(closed_eci),
                "open_model": matching_open.get("Model", "Unknown"),
                "open_date": matching_open["date"].isoformat(),
                "open_eci": float(matching_open["eci"]),
                "gap_months": round(gap_months, 1),
                "matched": True,
                "match_type": match_type,  # 'exact' or 'statistical'
            })
        else:
            # Unmatched closed model
            now = datetime.now()
            gap_days = (now - closed_date.to_pydatetime().replace(tzinfo=None)).days
            gap_months = gap_days / DAYS_PER_MONTH

            gaps.append({
                "closed_model": closed_row.get("Model", "Unknown"),
                "closed_date": closed_date.isoformat(),
                "closed_eci": float(closed_eci),
                "open_model": None,
                "open_date": None,
                "open_eci": None,
                "gap_months": round(gap_months, 1),
                "matched": False,
            })

    return gaps


def calculate_trends(df: pd.DataFrame) -> dict:
    """
    Calculate trends for frontier models before and after 2025.
    Returns:
      - Absolute Growth (Linear slope ECI/year)
      - Percentage Growth (Exponential fit)
      - Multiples per Year
      - Doubling Time
    """
    trends = {}
    
    cutoff = pd.Timestamp("2024-04-01")
    
    def get_stats(sub_df, name):
        if len(sub_df) < 2:
            return None
            
        # Prepare data
        dates_ordinal = sub_df["date"].map(datetime.toordinal).values
        ecis = sub_df["eci"].values
        
        # 1. Absolute Growth (Linear Regression: ECI ~ Date)
        lin_slope, lin_intercept, _, _, _ = linregress(dates_ordinal, ecis)
        yearly_absolute_growth = lin_slope * 365.25
        
        # Linear line points for plotting
        start_date_ord = dates_ordinal.min()
        end_date_ord = dates_ordinal.max()
        
        lin_start_eci = lin_slope * start_date_ord + lin_intercept
        lin_end_eci = lin_slope * end_date_ord + lin_intercept
        
        # 2. Exponential Growth (Linear Regression: ln(ECI) ~ Date)
        # Filter out non-positive ECIs if any (though ECI is usually > 0)
        valid_indices = ecis > 0
        if not np.any(valid_indices):
            return None
            
        log_ecis = np.log(ecis[valid_indices])
        log_dates = dates_ordinal[valid_indices]
        
        exp_slope, _, _, _, _ = linregress(log_dates, log_ecis)
        
        # Annual exponential rate constant (k in e^(kt))
        k_annual = exp_slope * 365.25
        
        # Metrics
        # Percentage Growth = (e^k - 1) * 100
        pct_growth = (np.exp(k_annual) - 1) * 100
        
        # Multiples per Year = e^k
        multiples_per_year = np.exp(k_annual)
        
        # Doubling Time = ln(2) / k
        doubling_time_years = np.log(2) / k_annual if k_annual > 0 else float('inf')

        return {
            "name": name,
            "absolute_growth_per_year": round(yearly_absolute_growth, 2),
            "percentage_growth_annualized": round(pct_growth, 1),
            "multiples_per_year": round(multiples_per_year, 2),
            "doubling_time_years": round(doubling_time_years, 2),
            "start_point": {
                "date": datetime.fromordinal(int(start_date_ord)).isoformat(),
                "eci": lin_start_eci
            },
            "end_point": {
                "date": datetime.fromordinal(int(end_date_ord)).isoformat(),
                "eci": lin_end_eci
            }
        }

    # Pre-2024
    pre_2024 = df[df["date"] < cutoff]
    trends["pre_apr_2024"] = get_stats(pre_2024, "Pre-Apr 2024")
    
    # Post-2024
    post_2024 = df[df["date"] >= cutoff]
    trends["post_apr_2024"] = get_stats(post_2024, "Post-Apr 2024")
    
    return trends


def _open_caught_up(
    open_score: float,
    open_std: float,
    sota_score: float,
    sota_std: float,
    threshold: float,
    z: float = Z_ONE_SIDED_05,
) -> bool:
    """Has the open model *plausibly caught up* to a historical SOTA model?

    Mirrors Epoch AI's bootstrap criterion (open beats SOTA in >=5% of paired
    bootstrap samples == SOTA not significantly better than open at the
    one-sided 5% level). With only point estimate + 90% CI per model, treat
    each bootstrap ECI as ~Normal(score, std**2) so the paired difference
    open - sota is Normal(open - sota, s_open^2 + s_sota^2); then

        P(open > sota) >= 0.05  <=>  sota - open <= z * SE,  SE = sqrt(s1^2+s2^2)

    (z = 1.645). Falls back to a point-estimate match within ``threshold`` when
    uncertainty is unavailable. See scripts/update_data.py for the full
    derivation and the independence caveat.
    """
    if (
        pd.notna(open_std)
        and pd.notna(sota_std)
        and (open_std > 0 or sota_std > 0)
    ):
        se = math.sqrt(open_std ** 2 + sota_std ** 2)
        return (sota_score - open_score) <= z * se
    return open_score >= sota_score - threshold


def calculate_gap_metrics(
    df: pd.DataFrame,
    score_col: str = "eci",
    threshold: float = ECI_MATCH_THRESHOLD,
    window_start: Optional[Any] = None,
    window_end: Optional[Any] = None,
    z: float = Z_ONE_SIDED_05,
) -> Optional[dict]:
    """Day-by-day open-vs-SOTA gap mirroring Epoch AI's methodology.

    Steps one day at a time across the analysis window. For each day the time
    (horizontal) gap is the months since the most recent historical
    closed-weight SOTA model the day's best open model has plausibly caught up
    to (see :func:`_open_caught_up`), with a strict variant requiring a strictly
    higher open point estimate; the vertical gap is the absolute SOTA ECI minus
    the best open ECI. Series are averaged across the window with 5th/95th
    percentile bands. Window defaults to the full overlapping history. This is a
    copy of the canonical implementation in scripts/update_data.py.
    """
    std_col = f"{score_col}_std"
    d = df.dropna(subset=["date", score_col]).copy()
    if d.empty or "Open" not in d.columns:
        return None
    d = d.sort_values("date", kind="mergesort")
    has_std = std_col in d.columns

    open_rows = d[d["Open"]]
    closed_rows = d[~d["Open"]]
    if open_rows.empty or closed_rows.empty:
        return None

    sota: list[dict] = []
    run_max = -np.inf
    for _, r in closed_rows.iterrows():
        s = r[score_col]
        if s > run_max:
            run_max = s
            sota.append({
                "date": r["date"],
                "score": float(s),
                "std": float(r[std_col]) if has_std and pd.notna(r.get(std_col)) else np.nan,
            })
    if not sota:
        return None

    first_open = open_rows["date"].min()
    first_closed = sota[0]["date"]
    ws = pd.Timestamp(window_start) if window_start is not None else max(first_open, first_closed)
    we = pd.Timestamp(window_end) if window_end is not None else d["date"].max()
    if ws > we:
        return None

    open_sorted = open_rows.sort_values("date", kind="mergesort")
    daily_time: list[float] = []
    daily_time_strict: list[float] = []
    daily_vertical: list[float] = []

    for day in pd.date_range(ws, we):
        open_avail = open_sorted[open_sorted["date"] <= day]
        if open_avail.empty:
            continue
        best_open = open_avail.loc[open_avail[score_col].idxmax()]
        best_open_score = float(best_open[score_col])
        best_open_std = (
            float(best_open[std_col])
            if has_std and pd.notna(best_open.get(std_col))
            else np.nan
        )

        sota_avail = [s for s in sota if s["date"] <= day]
        if not sota_avail:
            continue

        best_closed_score = max(s["score"] for s in sota_avail)
        absolute_sota = max(best_closed_score, best_open_score)
        daily_vertical.append(absolute_sota - best_open_score)

        ref_date = None
        for s in sorted(sota_avail, key=lambda x: x["date"], reverse=True):
            if _open_caught_up(best_open_score, best_open_std, s["score"], s["std"], threshold, z):
                ref_date = s["date"]
                break
        if ref_date is None:
            ref_date = sota_avail[0]["date"]
        daily_time.append((day - ref_date).days / DAYS_PER_MONTH)

        ref_date_strict = None
        for s in sorted(sota_avail, key=lambda x: x["date"], reverse=True):
            if best_open_score > s["score"]:
                ref_date_strict = s["date"]
                break
        if ref_date_strict is None:
            ref_date_strict = sota_avail[0]["date"]
        daily_time_strict.append((day - ref_date_strict).days / DAYS_PER_MONTH)

    if not daily_time or not daily_vertical:
        return None

    time_arr = np.array(daily_time)
    time_strict_arr = np.array(daily_time_strict)
    vert_arr = np.array(daily_vertical)
    t_lo, t_hi = np.quantile(time_arr, [0.05, 0.95])
    v_lo, v_hi = np.quantile(vert_arr, [0.05, 0.95])

    return {
        "avg_time_gap_months": float(np.mean(time_arr)),
        "avg_time_gap_months_strict": float(np.mean(time_strict_arr)),
        "time_gap_std": float(np.std(time_arr, ddof=1)) if len(time_arr) > 1 else 0.0,
        "time_gap_ci_90_low": float(t_lo),
        "time_gap_ci_90_high": float(t_hi),
        "avg_vertical_gap": float(np.mean(vert_arr)),
        "vertical_gap_ci_90_low": float(v_lo),
        "vertical_gap_ci_90_high": float(v_hi),
        "window_start": ws.isoformat(),
        "window_end": we.isoformat(),
        "n_days": int(len(time_arr)),
    }


def calculate_statistics(df: pd.DataFrame, gaps: list[dict]) -> dict:
    """Summary statistics: day-by-day open-vs-SOTA gap mirroring Epoch AI.

    The headline time gap and windowed vertical gap come from
    :func:`calculate_gap_metrics`; the matched/unmatched counts come from the
    per-closed-model ``gaps`` list.
    """
    df_open = df[df["Open"]].copy()
    df_closed = df[~df["Open"]].copy()

    matched_gaps = [g for g in gaps if g["matched"]]

    if len(df_open) == 0 or len(df_closed) == 0:
        return {
            "avg_horizontal_gap_months": 0,
            "avg_horizontal_gap_months_strict": 0,
            "std_horizontal_gap": 0,
            "ci_90_low": 0,
            "ci_90_high": 0,
            "current_vertical_gap": 0,
            "avg_vertical_gap": 0,
            "vertical_gap_ci_90_low": 0,
            "vertical_gap_ci_90_high": 0,
            "gap_window": None,
            "total_matched": len(matched_gaps),
            "total_unmatched": len(gaps) - len(matched_gaps),
        }

    metrics = calculate_gap_metrics(df, score_col="eci", threshold=ECI_MATCH_THRESHOLD)

    if metrics is not None:
        avg_gap = metrics["avg_time_gap_months"]
        avg_gap_strict = metrics["avg_time_gap_months_strict"]
        std_gap = metrics["time_gap_std"]
        ci_low = metrics["time_gap_ci_90_low"]
        ci_high = metrics["time_gap_ci_90_high"]
        avg_vertical = metrics["avg_vertical_gap"]
        vertical_ci_low = metrics["vertical_gap_ci_90_low"]
        vertical_ci_high = metrics["vertical_gap_ci_90_high"]
        gap_window = {
            "start": metrics["window_start"],
            "end": metrics["window_end"],
            "n_days": metrics["n_days"],
        }
    else:
        avg_gap = avg_gap_strict = std_gap = ci_low = ci_high = 0
        avg_vertical = vertical_ci_low = vertical_ci_high = 0
        gap_window = None

    # Current ("vertical") gap snapshot kept alongside the windowed average.
    best_open_eci = df_open["eci"].max() if len(df_open) > 0 else 0
    best_closed_eci = df_closed["eci"].max() if len(df_closed) > 0 else 0
    vertical_gap = best_closed_eci - best_open_eci

    return {
        "avg_horizontal_gap_months": round(avg_gap, 1),
        "avg_horizontal_gap_months_strict": round(avg_gap_strict, 1),
        "std_horizontal_gap": round(std_gap, 1),
        "ci_90_low": round(ci_low, 1),
        "ci_90_high": round(ci_high, 1),
        "current_vertical_gap": round(vertical_gap, 1),
        "avg_vertical_gap": round(avg_vertical, 1),
        "vertical_gap_ci_90_low": round(vertical_ci_low, 1),
        "vertical_gap_ci_90_high": round(vertical_ci_high, 1),
        "gap_window": gap_window,
        "total_matched": len(matched_gaps),
        "total_unmatched": len(gaps) - len(matched_gaps),
    }


@app.route("/")
def index():
    """Serve the main visualization page."""
    return render_template("index.html")


@app.route("/data.json")
def data_json():
    """Serve the prebuilt data.json from the project root so the static
    frontend (which fetches `data.json` relative to the page, matching the
    GitHub Pages deployment) works under Flask too."""
    return send_from_directory(Path(__file__).parent, "data.json")


@app.route("/api/data")
def api_data():
    """Return processed ECI data as JSON."""
    try:
        data = process_data()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
