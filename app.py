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
from typing import Any

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template
from scipy.stats import linregress, norm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DAYS_PER_MONTH = 365.25 / 12  # 30.4375 - accurate average days per month

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


def calculate_statistics(df: pd.DataFrame, gaps: list[dict]) -> dict:
    """
    Calculate summary statistics using the notebook's sampling approach:
    Sample 100 ECI levels uniformly and calculate gaps at each level.
    """
    df_open = df[df["Open"]].copy()
    df_closed = df[~df["Open"]].copy()
    
    # Get ECI range with valid horizontal gaps (overlapping range)
    if len(df_open) == 0 or len(df_closed) == 0:
        return {
            "avg_horizontal_gap_months": 0,
            "std_horizontal_gap": 0,
            "ci_90_low": 0,
            "ci_90_high": 0,
            "current_vertical_gap": 0,
            "total_matched": 0,
            "total_unmatched": len(gaps),
        }
    
    start_eci = max(df_open["eci"].min(), df_closed["eci"].min())
    end_eci = min(df_open["eci"].max(), df_closed["eci"].max())  # Use min to avoid censored observations
    
    horizontal_gaps = []
    
    # Keep track of which open models might still qualify (optimization from notebook)
    df_open_possible = df_open.sort_values("date").copy()
    
    # Sample 100 ECI levels uniformly across the valid range
    for cur_eci in np.linspace(start_eci, end_eci, 100):
        # Find earliest closed model with ECI >= cur_eci
        closed_candidates = df_closed[df_closed["eci"] >= cur_eci].sort_values("date")
        if len(closed_candidates) == 0:
            continue
        cur_closed_model = closed_candidates.iloc[0]
        
        # Find first open model that matches (per notebook logic)
        cur_open_model = None
        for _, row in df_open_possible.iterrows():
            if pd.isna(row["eci"]) or pd.isna(row["date"]):
                continue
                
            if row["eci"] >= cur_eci - 1:
                cur_open_model = row
                gap = (cur_open_model["date"] - cur_closed_model["date"]).days / DAYS_PER_MONTH
                horizontal_gaps.append(gap)
                break
            else:
                pass
        
        # If loop finished with no match, use current date
        if cur_open_model is None:
            now = datetime.now()
            # Calculate gap from closed model release to now
            # Only count if "now" is after release date (should be always true for valid data)
            gap = (now - cur_closed_model["date"].to_pydatetime().replace(tzinfo=None)).days / DAYS_PER_MONTH
            horizontal_gaps.append(gap)
    
    # Calculate statistics from sampled gaps
    if horizontal_gaps:
        avg_gap = np.mean(horizontal_gaps)
        std_gap = np.std(horizontal_gaps, ddof=1)  # Unbiased estimator
        n = len(horizontal_gaps)
        # 90% confidence interval on the mean (z = 1.645 for 90% CI)
        sem = std_gap / np.sqrt(n)  # Standard error of the mean
        ci_low = avg_gap - 1.645 * sem
        ci_high = avg_gap + 1.645 * sem
    else:
        avg_gap = std_gap = ci_low = ci_high = 0

    # Calculate current vertical gap (difference in best ECI scores)
    best_open_eci = df_open["eci"].max() if len(df_open) > 0 else 0
    best_closed_eci = df_closed["eci"].max() if len(df_closed) > 0 else 0
    vertical_gap = best_closed_eci - best_open_eci

    matched_gaps = [g for g in gaps if g["matched"]]
    
    return {
        "avg_horizontal_gap_months": round(avg_gap, 1),
        "std_horizontal_gap": round(std_gap, 1),
        "ci_90_low": round(ci_low, 1),
        "ci_90_high": round(ci_high, 1),
        "current_vertical_gap": round(vertical_gap, 1),
        "total_matched": len(matched_gaps),
        "total_unmatched": len(gaps) - len(matched_gaps),
    }


@app.route("/")
def index():
    """Serve the main visualization page."""
    return render_template("index.html")


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
