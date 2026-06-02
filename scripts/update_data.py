
import csv
import io
import json
import logging
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from scipy.stats import linregress, norm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
# benchmarked_models.csv ships one row per (Model, variant) — e.g. GPT-5.5
# with reasoning_effort=low/medium/high/xhigh — and includes Organization,
# Version release date, Model accessibility, and ECI inline. We aggregate
# to one row per Model (best variant by ECI). This replaces eci_scores.csv,
# which currently ships Organization/date/accessibility as all-NaN.
BENCHMARKED_MODELS_URL = "https://epoch.ai/data/benchmarked_models.csv"
# Fallback metadata source for the rare row missing Organization or date.
ALL_AI_MODELS_URL = "https://epoch.ai/data/all_ai_models.csv"
# epoch.ai blocks the default urllib User-Agent with 403.
EPOCH_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
ECI_MATCH_THRESHOLD = 1.0  # ECI points - model is "matched" if within this range
DAYS_PER_MONTH = 365.25 / 12  # 30.4375 - accurate average days per month accounting for leap years

# Labs that ship ~exclusively closed weights. A blank Model accessibility from
# one of these is a data gap, not an implicit open release, so we don't flip
# it to open. Meta/Mistral/Microsoft are excluded because they release open
# models often enough that a blank could plausibly be open.
KNOWN_CLOSED_LABS = {
    "OpenAI",
    "Anthropic",
    "Google",
    "Google DeepMind",
    "xAI",
}


def _is_known_closed_lab(org: str) -> bool:
    if not isinstance(org, str):
        return False
    org_lower = org.lower()
    return any(lab.lower() in org_lower for lab in KNOWN_CLOSED_LABS)

# Import CSV-based benchmark fetcher (no credentials required)
try:
    from csv_benchmark_fetcher import CSVBenchmarkFetcher, BENCHMARK_CSV_CONFIG
    CSV_FETCHER_AVAILABLE = True
except ImportError as e:
    CSV_FETCHER_AVAILABLE = False
    BENCHMARK_CSV_CONFIG = {}
    print(f"Note: CSV benchmark fetcher unavailable ({e})")

# Legacy Airtable fetcher (optional, requires credentials)
try:
    from benchmark_fetcher import BenchmarkDataFetcher, BENCHMARK_CONFIG
    AIRTABLE_FETCHER_AVAILABLE = True
except Exception as e:
    AIRTABLE_FETCHER_AVAILABLE = False
    BENCHMARK_CONFIG = {}
    # Only log if CSV fetcher also unavailable
    if not CSV_FETCHER_AVAILABLE:
        print(f"Note: Airtable benchmark fetcher unavailable ({type(e).__name__}: {e})")

# Import METR fetcher
try:
    from metr_fetcher import fetch_metr_data
    METR_FETCHER_AVAILABLE = True
except ImportError as e:
    METR_FETCHER_AVAILABLE = False
    print(f"Note: METR benchmark unavailable ({e})")

def get_rank(
    df: pd.DataFrame,
    n: int | None = None,
    sort_col: str = "date",
    val_col: str = "eci",
) -> pd.Series:
    """
    Cumulative rank of *val_col* up to each row, ordered by *sort_col*.
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

def _fetch_epoch_csv(url: str, timeout: int = 30) -> pd.DataFrame:
    """GET a CSV from epoch.ai with a browser UA (server 403s on urllib).

    Parses via stdlib csv.DictReader, which correctly handles the quoted
    multi-line Description fields in benchmarked_models.csv that both
    pandas parser engines mistokenize.
    """
    response = requests.get(url, headers={"User-Agent": EPOCH_UA}, timeout=timeout)
    response.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(response.text)))
    return pd.DataFrame(rows)


def _strip_date_suffix(name: str) -> str:
    """Turn "Gemini 2.5 Flash (Sep 2025)" into "Gemini 2.5 Flash"."""
    return re.sub(r"\s*\([^)]*\d{4}\)\s*$", "", str(name)).strip()


def _aggregate_benchmarked_models(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse benchmarked_models.csv to one row per Model.

    The CSV ships one row per (Model, variant) — e.g. GPT-5.5 with
    reasoning_effort low/medium/high/xhigh, plus pre-release duplicates.
    We keep the row with the highest ECI per Model (the best variant on
    the leaderboard) and rename Version release date → date so downstream
    code that expects an eci_scores.csv-shaped frame keeps working.
    """
    df = df.copy()
    # csv.DictReader returns everything as strings; coerce the numeric
    # columns we use downstream.
    for col in ("eci", "eci_ci_low", "eci_ci_high"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Version release date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Version release date": "date"})

    # Drop rows without an ECI score (variants pending evaluation).
    df = df.dropna(subset=["eci"])

    # For each Model, keep the variant with the highest ECI.
    df = df.sort_values("eci", ascending=False, kind="mergesort")
    df = df.drop_duplicates(subset=["Model"], keep="first").reset_index(drop=True)
    return df


def _backfill_from_all_ai_models(df: pd.DataFrame) -> pd.DataFrame:
    """Fill Organization, Country, and date from all_ai_models.csv.

    Used as a defensive fallback for the rare model in benchmarked_models.csv
    that's missing Organization or Version release date. Joins on Model name
    with a date-suffix-stripped fallback.
    """
    try:
        models = _fetch_epoch_csv(ALL_AI_MODELS_URL, timeout=60)
    except Exception as e:
        logger.warning(f"Could not fetch all_ai_models.csv for backfill: {e}")
        return df

    needed = ["Model", "Organization", "Country (of organization)", "Publication date"]
    if not all(c in models.columns for c in needed):
        logger.warning("all_ai_models.csv missing expected columns; skipping backfill")
        return df

    models = models[needed].copy()
    models["Model"] = models["Model"].astype(str).str.strip()
    models["_stripped"] = models["Model"].apply(_strip_date_suffix)

    exact = (models.drop_duplicates("Model")
                   .set_index("Model")[["Organization", "Country (of organization)", "Publication date"]])
    stripped = (models.drop_duplicates("_stripped")
                      .set_index("_stripped")[["Organization", "Country (of organization)", "Publication date"]])

    df = df.copy()
    df["Model"] = df["Model"].astype(str).str.strip()
    keys_stripped = df["Model"].apply(_strip_date_suffix)

    for src_col, dst_col in (
        ("Organization", "Organization"),
        ("Country (of organization)", "Country (of organization)"),
        ("Publication date", "date"),
    ):
        if dst_col not in df.columns:
            df[dst_col] = pd.NA
        exact_vals = df["Model"].map(exact[src_col])
        stripped_vals = keys_stripped.map(stripped[src_col])
        backfill = exact_vals.combine_first(stripped_vals)
        df[dst_col] = df[dst_col].combine_first(backfill)

    matched = df["Organization"].notna().sum()
    logger.info(f"  Backfilled Organization for {matched}/{len(df)} models from all_ai_models.csv")
    return df


def fetch_eci_data() -> pd.DataFrame:
    """Fetch ECI scores from Epoch AI."""
    logger.info("Fetching data from Epoch AI (benchmarked_models.csv)...")
    try:
        df = _fetch_epoch_csv(BENCHMARKED_MODELS_URL)

        # benchmarked_models.csv has one row per (Model, variant). Keep the
        # best variant per Model and rename Version release date → date.
        df = _aggregate_benchmarked_models(df)

        # Defensive fallback: fill any still-missing Organization / date
        # from all_ai_models.csv (rare; benchmarked_models.csv is usually
        # complete on its own).
        df = _backfill_from_all_ai_models(df)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Any row the backfill couldn't resolve (or any other expected-
        # string column that's still NaN) gets "Unknown" so downstream
        # `.str.contains` calls work and JSON never sees pd.NA.
        for col in ("Model", "Display name", "Organization",
                    "Country (of organization)"):
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)

        # Blank Model accessibility is treated as open *unless* the row is from
        # a known closed-weights lab (OpenAI etc.), where a blank is almost
        # certainly a data gap rather than an implicit open release. csv.DictReader
        # yields "" for empty cells (not NaN), so a plain fillna above would miss
        # them and process_data would silently classify everything blank as closed.
        if "Model accessibility" in df.columns:
            accessibility = df["Model accessibility"].fillna("").astype(str).str.strip()
            is_blank = accessibility == ""
            org = df.get("Organization", pd.Series([""] * len(df), index=df.index)).astype(str)
            from_closed_lab = org.apply(_is_known_closed_lab)
            df["Model accessibility"] = (
                accessibility
                .mask(is_blank & from_closed_lab, "Unknown")
                .mask(is_blank & ~from_closed_lab, "Open weights (assumed)")
            )

        # Derive eci_std from confidence intervals (assuming 90% CI)
        # For 90% CI, z = 1.645, so std = (ci_high - ci_low) / (2 * 1.645)
        if "eci_ci_low" in df.columns and "eci_ci_high" in df.columns:
            df["eci_std"] = (df["eci_ci_high"] - df["eci_ci_low"]) / (2 * 1.645)
        else:
            df["eci_std"] = pd.NA
        
        return df
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise

def calculate_horizontal_gaps(
    df: pd.DataFrame,
    score_col: str = "eci",
    threshold: float = ECI_MATCH_THRESHOLD,
    model_col: str = "Model"
) -> list[dict]:
    """
    Calculate horizontal gaps between closed and open models.

    Args:
        df: DataFrame with model data
        score_col: Column name for the score/metric
        threshold: Gap matching threshold (additive: score >= closed - threshold)
        model_col: Column name for model identifier
    """
    if len(df) == 0:
        return []

    std_col = f"{score_col}_std"
    has_std = std_col in df.columns

    df_open = df[df["Open"]].sort_values("date")
    df_closed = df[~df["Open"]].sort_values("date")

    gaps = []

    for _, closed_row in df_closed.iterrows():
        closed_score = closed_row[score_col]
        closed_date = closed_row["date"]
        closed_std = closed_row[std_col] if has_std else np.nan

        if pd.isna(closed_score) or pd.isna(closed_date):
            continue

        matching_open = None
        match_type = None
        for _, open_row in df_open.iterrows():
            if pd.isna(open_row[score_col]) or pd.isna(open_row["date"]):
                continue

            if open_row["date"] <= closed_date:
                continue

            # Match when the open model has plausibly caught up to the closed
            # model under the within-5% bootstrap criterion (falls back to the
            # threshold when no CI is available); see _open_caught_up.
            open_std = open_row[std_col] if has_std else np.nan
            if _open_caught_up(open_row[score_col], open_std, closed_score, closed_std, threshold):
                matching_open = open_row
                match_type = "caught_up"
                break

        if matching_open is not None:
            gap_days = (matching_open["date"] - closed_date).days
            gap_months = gap_days / DAYS_PER_MONTH

            gaps.append({
                "closed_model": closed_row.get(model_col, closed_row.get("model", "Unknown")),
                "closed_date": closed_date.isoformat(),
                "closed_score": float(closed_score),
                "open_model": matching_open.get(model_col, matching_open.get("model", "Unknown")),
                "open_date": matching_open["date"].isoformat(),
                "open_score": float(matching_open[score_col]),
                "gap_months": round(gap_months, 1),
                "matched": True,
                "match_type": match_type,
            })
        else:
            now = datetime.now()
            gap_days = (now - closed_date.to_pydatetime().replace(tzinfo=None)).days
            gap_months = gap_days / DAYS_PER_MONTH

            gaps.append({
                "closed_model": closed_row.get(model_col, closed_row.get("model", "Unknown")),
                "closed_date": closed_date.isoformat(),
                "closed_score": float(closed_score),
                "open_model": None,
                "open_date": None,
                "open_score": None,
                "gap_months": round(gap_months, 1),
                "matched": False,
            })

    return gaps

def calculate_trends(df: pd.DataFrame, score_col: str = "eci", use_apr_2024_split: bool = False) -> dict:
    """
    Calculate trends dynamically based on the data's date range.

    For datasets spanning > 1 year, splits at the midpoint to show early vs recent trends.
    For shorter datasets, shows only the overall trend.

    Args:
        df: DataFrame with 'date' and score columns
        score_col: Name of the score column
        use_apr_2024_split: If True, use April 2024 as split point (for ECI specifically)
    """
    trends = {}

    if len(df) < 2:
        return {"overall": None, "metadata": {"has_split": False}}

    df = df.sort_values("date")
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range_days = (max_date - min_date).days

    def get_stats(sub_df, name):
        if len(sub_df) < 2:
            return None

        dates_ordinal = sub_df["date"].map(datetime.toordinal).values
        scores = sub_df[score_col].values

        # Check if all dates are identical (can't compute regression)
        if len(set(dates_ordinal)) < 2:
            return None

        lin_slope, lin_intercept, _, _, _ = linregress(dates_ordinal, scores)
        yearly_absolute_growth = lin_slope * 365.25

        start_date_ord = dates_ordinal.min()
        end_date_ord = dates_ordinal.max()

        lin_start_score = lin_slope * start_date_ord + lin_intercept
        lin_end_score = lin_slope * end_date_ord + lin_intercept

        valid_indices = scores > 0
        if not np.any(valid_indices):
            return None

        log_scores = np.log(scores[valid_indices])
        log_dates = dates_ordinal[valid_indices]

        # Check if we have enough unique dates for exponential regression
        if len(set(log_dates)) < 2:
            # Can't compute exponential growth, use defaults
            pct_growth = 0
            multiples_per_year = 1
            doubling_time_years = None  # Use None instead of inf for JSON compatibility
        else:
            exp_slope, _, _, _, _ = linregress(log_dates, log_scores)
            k_annual = exp_slope * 365.25
            pct_growth = (np.exp(k_annual) - 1) * 100
            multiples_per_year = np.exp(k_annual)
            doubling_time_years = np.log(2) / k_annual if k_annual > 0 else None

        return {
            "name": name,
            "absolute_growth_per_year": round(yearly_absolute_growth, 2),
            "percentage_growth_annualized": round(pct_growth, 1),
            "multiples_per_year": round(multiples_per_year, 2),
            "doubling_time_years": round(doubling_time_years, 2) if doubling_time_years is not None else None,
            "start_point": {
                "date": datetime.fromordinal(int(start_date_ord)).isoformat(),
                "eci": lin_start_score  # Keep as "eci" for frontend compatibility
            },
            "end_point": {
                "date": datetime.fromordinal(int(end_date_ord)).isoformat(),
                "eci": lin_end_score  # Keep as "eci" for frontend compatibility
            }
        }

    def format_date_label(dt):
        """Format date as 'Mon YYYY' for trend labels."""
        return dt.strftime("%b %Y")

    # Calculate overall trend
    trends["overall"] = get_stats(df, "Overall")

    # Only split if we have > 1 year of data and enough points
    if date_range_days > 365 and len(df) >= 6:
        # Find midpoint date
        midpoint = min_date + pd.Timedelta(days=date_range_days // 2)

        early_df = df[df["date"] < midpoint]
        recent_df = df[df["date"] >= midpoint]

        # Only create split trends if both periods have enough data
        if len(early_df) >= 3 and len(recent_df) >= 3:
            early_label = f"Pre-{format_date_label(midpoint)}"
            recent_label = f"Post-{format_date_label(midpoint)}"

            trends["early"] = get_stats(early_df, early_label)
            trends["recent"] = get_stats(recent_df, recent_label)
            trends["metadata"] = {
                "has_split": True,
                "split_date": midpoint.isoformat(),
                "split_label": format_date_label(midpoint),
                "early_key": "early",
                "recent_key": "recent"
            }
        else:
            trends["metadata"] = {"has_split": False}
    else:
        trends["metadata"] = {"has_split": False}

    # For ECI specifically, use April 2024 as the split point (meaningful for AI progress)
    if use_apr_2024_split and date_range_days > 365:
        apr_2024 = pd.Timestamp("2024-04-01")
        if min_date < apr_2024 < max_date:
            pre_apr = df[df["date"] < apr_2024]
            post_apr = df[df["date"] >= apr_2024]
            if len(pre_apr) >= 2 and len(post_apr) >= 2:
                trends["pre_apr_2024"] = get_stats(pre_apr, "Pre-Apr 2024")
                trends["post_apr_2024"] = get_stats(post_apr, "Post-Apr 2024")
                # Override the dynamic split with April 2024 for ECI
                trends["metadata"] = {
                    "has_split": True,
                    "split_date": apr_2024.isoformat(),
                    "split_label": "Apr 2024",
                    "early_key": "pre_apr_2024",
                    "recent_key": "post_apr_2024"
                }

    return trends

def estimate_current_gap(gaps: list[dict], matched_gaps_months: list[float], use_survival_analysis: bool = True, prior_from_first_match: bool = False) -> dict:
    """
    Estimate the current gap using unmatched models.

    When use_survival_analysis=True (default), fits a log-normal distribution
    to historical matched gaps and uses it to estimate expected gap for
    censored (unmatched) observations.

    When prior_from_first_match=True, the prior is fitted only to matched
    gaps where the closed model was released after the first match event.
    This avoids contaminating the prior with gaps from an era when no
    open-source competitor existed.

    When use_survival_analysis=False, the estimate is simply the age of the
    oldest unmatched model (the known minimum gap).
    """
    unmatched = [g for g in gaps if not g["matched"]]

    if not unmatched:
        return {
            "estimated_current_gap": 0,
            "min_current_gap": 0,
            "confidence": "high",
            "unmatched_ages": [],
            "method": "no_unmatched",
        }

    unmatched_ages = sorted([g["gap_months"] for g in unmatched], reverse=True)
    oldest_unmatched = max(unmatched_ages)

    if not use_survival_analysis:
        return {
            "estimated_current_gap": round(float(oldest_unmatched), 1),
            "min_current_gap": round(float(oldest_unmatched), 1),
            "confidence": "high" if len(unmatched) >= 2 else "medium",
            "unmatched_ages": [round(a, 1) for a in unmatched_ages],
            "method": "oldest_unmatched",
        }

    # Optionally filter the prior to only gaps from the competitive era
    if prior_from_first_match:
        matched = [g for g in gaps if g["matched"] and g.get("open_date")]
        if matched:
            first_match_date = min(g["open_date"] for g in matched)
            matched_gaps_months = [
                g["gap_months"] for g in matched
                if g["closed_date"] >= first_match_date
            ]

    if not matched_gaps_months or len(matched_gaps_months) < 3:
        # Use 75th percentile of available data if possible, otherwise conservative estimate
        if matched_gaps_months and len(matched_gaps_months) >= 1:
            historical_p75 = np.percentile(matched_gaps_months, 75)
            estimated = max(oldest_unmatched, historical_p75)
        else:
            # No historical data - use 50% buffer as conservative estimate
            estimated = oldest_unmatched * 1.5
        return {
            "estimated_current_gap": round(estimated, 1),
            "min_current_gap": round(float(oldest_unmatched), 1),
            "confidence": "low",
            "unmatched_ages": [round(a, 1) for a in unmatched_ages],
            "method": "insufficient_data_heuristic",
        }

    # Fit log-normal distribution to matched gaps
    matched_positive = [g for g in matched_gaps_months if g > 0]
    if len(matched_positive) < 3:
        # Use 75th percentile of positive data if available
        if matched_positive:
            historical_p75 = np.percentile(matched_positive, 75)
            estimated = max(oldest_unmatched, historical_p75)
        else:
            estimated = oldest_unmatched * 1.5
        return {
            "estimated_current_gap": round(estimated, 1),
            "min_current_gap": round(float(oldest_unmatched), 1),
            "confidence": "low",
            "unmatched_ages": [round(a, 1) for a in unmatched_ages],
            "method": "insufficient_positive_data",
        }

    log_matched = np.log(matched_positive)
    mu_prior = np.mean(log_matched)
    sigma_prior = np.std(log_matched, ddof=1)  # Use unbiased estimator (Bessel's correction)

    if sigma_prior < 0.1:
        # Very low variance means all historical gaps are nearly identical
        # Use a small but reasonable sigma based on coefficient of variation ~10%
        sigma_prior = 0.1  # Corresponds to ~10% CV in log-normal

    def expected_given_greater_than(c, mu, sigma):
        """E[X | X > c] for log-normal distribution."""
        if c <= 0:
            return np.exp(mu + sigma**2 / 2)

        log_c = np.log(c)
        z = (log_c - mu) / sigma
        survival = 1 - norm.cdf(z)

        if survival < 1e-10:
            return c * 2

        z_shifted = (mu + sigma**2 - log_c) / sigma
        return np.exp(mu + sigma**2 / 2) * norm.cdf(z_shifted) / survival

    expected_gaps = []
    for age in unmatched_ages:
        if age > 0:
            expected_gaps.append(expected_given_greater_than(age, mu_prior, sigma_prior))

    if expected_gaps:
        weights = np.array(unmatched_ages[:len(expected_gaps)])
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
        estimated = np.average(expected_gaps, weights=weights)
        estimated = max(estimated, oldest_unmatched)

        prior_mean = np.exp(mu_prior + sigma_prior**2 / 2)
        deviation_ratio = oldest_unmatched / prior_mean if prior_mean > 0 else 2
        if deviation_ratio < 1.5:
            confidence = "high"
        elif deviation_ratio < 2.5:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        estimated = oldest_unmatched * 1.3
        confidence = "low"

    return {
        "estimated_current_gap": round(float(estimated), 1),
        "min_current_gap": round(float(oldest_unmatched), 1),
        "confidence": confidence,
        "unmatched_ages": [round(a, 1) for a in unmatched_ages],
        "method": "survival_analysis_mle",
        "prior_from_first_match": prior_from_first_match,
        "prior_params": {
            "mu": round(float(mu_prior), 3),
            "sigma": round(float(sigma_prior), 3),
            "prior_mean_months": round(float(np.exp(mu_prior + sigma_prior**2 / 2)), 1),
        },
    }


def calculate_historical_gaps(
    df: pd.DataFrame,
    score_col: str = "eci",
    threshold: float = ECI_MATCH_THRESHOLD,
    model_col: str = "Model"
) -> list[dict]:
    """
    Gap metric sampled monthly through history, using the SAME within-5%
    catch-up criterion as calculate_gap_metrics (see _open_caught_up).

    At each monthly checkpoint we take the best open-weight model available and
    find the most recent closed-weight SOTA model it has plausibly caught up to;
    the gap is the time elapsed since that SOTA model. ``matched`` indicates the
    open frontier has caught up to the *current* closed frontier. This is the
    monthly-sampled companion to the day-by-day headline statistic, so the chart
    and the headline tell a consistent story.
    """
    std_col = f"{score_col}_std"
    has_std = std_col in df.columns
    df = df.dropna(subset=["date", score_col]).copy().sort_values("date", kind="mergesort")

    if len(df) < 2:
        return []

    df_open_all = df[df["Open"]]
    df_closed_all = df[~df["Open"]]
    if df_open_all.empty or df_closed_all.empty:
        return []

    # Closed-weight SOTA sequence (running max), carrying model name + std.
    sota: list[dict] = []
    run_max = -np.inf
    for _, r in df_closed_all.iterrows():
        s = r[score_col]
        if s > run_max:
            run_max = s
            sota.append({
                "date": r["date"],
                "score": float(s),
                "std": float(r[std_col]) if has_std and pd.notna(r.get(std_col)) else np.nan,
                "model": r.get(model_col, r.get("model", "Unknown")),
            })
    if not sota:
        return []

    historical_gaps = []
    min_date = df["date"].min()
    max_date = pd.Timestamp(datetime.now())
    current = pd.Timestamp(min_date) + pd.DateOffset(months=6)  # Start 6 months in

    while current <= max_date:
        open_avail = df_open_all[df_open_all["date"] <= current]
        sota_avail = [s for s in sota if s["date"] <= current]
        if open_avail.empty or not sota_avail:
            current += pd.DateOffset(months=1)
            continue

        best_open = open_avail.loc[open_avail[score_col].idxmax()]
        best_open_score = float(best_open[score_col])
        best_open_std = (
            float(best_open[std_col])
            if has_std and pd.notna(best_open.get(std_col))
            else np.nan
        )

        # Most recent SOTA the open frontier has plausibly caught up to.
        ref = None
        for s in sorted(sota_avail, key=lambda x: x["date"], reverse=True):
            if _open_caught_up(best_open_score, best_open_std, s["score"], s["std"], threshold):
                ref = s
                break
        if ref is None:
            ref = sota_avail[0]
        gap_months = max(0.0, (current - ref["date"]).days / DAYS_PER_MONTH)

        # Current closed frontier (highest-scoring SOTA so far) for the matched flag.
        top_sota = max(sota_avail, key=lambda x: x["score"])
        is_matched = _open_caught_up(
            best_open_score, best_open_std, top_sota["score"], top_sota["std"], threshold
        )

        historical_gaps.append({
            "date": current.isoformat(),
            "gap_months": round(float(gap_months), 1),
            "matched": bool(is_matched),
            "reference_model": top_sota["model"],
            "reference_score": round(float(top_sota["score"]), 1),
            "open_frontier_model": best_open.get(model_col, best_open.get("model", "Unknown")),
            "open_frontier_score": round(float(best_open_score), 1),
        })

        current += pd.DateOffset(months=1)

    return historical_gaps


# One-sided 5% critical value: P(Z <= 1.6448) = 0.95. The bootstrap
# ">=5% of paired samples" rule is the one-sided 5% significance test in
# disguise (see _open_caught_up).
Z_ONE_SIDED_05 = float(norm.ppf(0.95))  # 1.6448...


def _open_caught_up(
    open_score: float,
    open_std: float,
    sota_score: float,
    sota_std: float,
    threshold: float,
    z: float = Z_ONE_SIDED_05,
) -> bool:
    """Has the open model *plausibly caught up* to a historical SOTA model?

    Epoch AI's published methodology resamples the underlying benchmark scores
    with replacement, refits the ECI model on each resample, and (preserving
    the pairing between models across resamples) counts a prior SOTA model as
    *not significantly better* than the open model when the open model's
    bootstrapped ECI exceeds the SOTA model's in at least 5% of paired samples.

    We only have the published point estimate and 90% CI per model, not the
    joint bootstrap draws, so we mirror the test analytically. Treating each
    model's bootstrap ECI as ~Normal(score, std**2) (std recovered from the
    90% CI as (ci_high - ci_low) / (2 * 1.645)), the paired difference
    D = open - sota is Normal(open - sota, s_open**2 + s_sota**2), so

        P(open > sota) = Phi((open - sota) / SE),  SE = sqrt(s_open^2 + s_sota^2)

    and therefore

        P(open > sota) >= 0.05  <=>  sota - open <= z * SE   (z = 1.645).

    That right-hand inequality is exactly "the SOTA model is not significantly
    better than the open model at the one-sided 5% level," which is the
    criterion Epoch describes.

    Caveat: Epoch preserves the pairing between models across bootstrap
    samples, which induces positive correlation and shrinks Var(D). Using the
    independent marginals here overstates SE, making this test marginally more
    permissive (time gaps marginally smaller) than the true paired bootstrap.
    With only marginal CIs published, independence is the closest available
    mirror.

    Falls back to a point-estimate match within ``threshold`` when either
    model lacks usable uncertainty (e.g. METR horizons, which ship no CI).
    """
    if (
        pd.notna(open_std)
        and pd.notna(sota_std)
        and (open_std > 0 or sota_std > 0)
    ):
        se = math.sqrt(open_std ** 2 + sota_std ** 2)
        return (sota_score - open_score) <= z * se
    # No usable uncertainty: fall back to "approximately equal" within threshold.
    return open_score >= sota_score - threshold


def calculate_gap_metrics(
    df: pd.DataFrame,
    score_col: str = "eci",
    threshold: float = ECI_MATCH_THRESHOLD,
    window_start: Optional[Any] = None,
    window_end: Optional[Any] = None,
    z: float = Z_ONE_SIDED_05,
) -> Optional[dict]:
    """Day-by-day open-vs-SOTA gap, mirroring Epoch AI's methodology.

    Proceeds one day at a time across the analysis window. For each day:

    * **Time (horizontal) gap** = days elapsed since the *most recent*
      historical closed-weight SOTA model that the day's best open-weight model
      has plausibly caught up to (see :func:`_open_caught_up`). We also compute
      a *strict* variant that requires the open model's point estimate to be
      strictly higher than the SOTA model it is catching up to.
    * **Vertical (ECI) gap** = absolute SOTA ECI available that day minus the
      best open-weight ECI available that day. Release dates carry no
      uncertainty, so this is a plain difference.

    Both series are averaged across the window's days; we report the 5th/95th
    percentiles as a 90% interval (the dispersion of the daily gap, matching the
    reference notebook, not a standard error of the mean).

    ``df`` is expected to be the per-group frontier (open and closed running-max
    records). The closed records form the historical SOTA sequence and the open
    records define the open frontier; we recompute the closed running max
    defensively so the function is correct for any closed input. The window
    defaults to the full overlapping history -- from the first day both an open
    and a closed model exist, through the latest model date -- and is
    configurable (e.g. pass 2026-01-01 / 2026-05-28 to reproduce a fixed-window
    figure).
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

    # Historical closed-weight SOTA sequence: closed models that set a new
    # closed-weight record, monotonically increasing in score by date.
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
    # Default the window end to "today" (not the latest release date) so the
    # gap keeps growing while no new open model has caught up, mirroring
    # calculate_historical_gaps and the current-gap framing.
    we = pd.Timestamp(window_end) if window_end is not None else pd.Timestamp(datetime.now())
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

        # Vertical gap: absolute SOTA (best closed or best open so far) minus
        # the open frontier. Open lags closed in this data, so this is just
        # best_closed - best_open, but max() keeps it correct (>= 0) if an open
        # model ever leads.
        best_closed_score = max(s["score"] for s in sota_avail)
        absolute_sota = max(best_closed_score, best_open_score)
        daily_vertical.append(absolute_sota - best_open_score)

        # Time gap (lenient / bootstrap criterion): most recent SOTA model the
        # open model has plausibly caught up to.
        ref_date = None
        for s in sorted(sota_avail, key=lambda x: x["date"], reverse=True):
            if _open_caught_up(best_open_score, best_open_std, s["score"], s["std"], threshold, z):
                ref_date = s["date"]
                break
        if ref_date is None:
            # Open hasn't caught even the earliest SOTA: lag is the full history.
            ref_date = sota_avail[0]["date"]
        daily_time.append((day - ref_date).days / DAYS_PER_MONTH)

        # Time gap (strict): most recent SOTA strictly below open's point estimate.
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


def calculate_statistics(
    df: pd.DataFrame,
    gaps: list[dict],
    score_col: str = "eci",
    use_survival_analysis: bool = True,
    prior_from_first_match: bool = False,
    threshold: float = ECI_MATCH_THRESHOLD,
    window_start: Optional[Any] = None,
    window_end: Optional[Any] = None,
) -> dict:
    """Calculate summary statistics.

    The headline time gap and vertical gap come from
    :func:`calculate_gap_metrics`, which steps day-by-day across the analysis
    window mirroring Epoch AI's open-vs-SOTA methodology. The per-closed-model
    ``gaps`` list (matched/unmatched counts) and the survival-based current-gap
    estimate are computed separately and left unchanged.
    """
    df_open = df[df["Open"]].copy()
    df_closed = df[~df["Open"]].copy()

    matched_gaps = [g for g in gaps if g["matched"]]
    matched_gaps_months = [g["gap_months"] for g in matched_gaps]

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
            "current_gap_estimate": {
                "estimated_current_gap": 0,
                "min_current_gap": 0,
                "confidence": "low",
                "unmatched_ages": [],
            },
        }

    metrics = calculate_gap_metrics(
        df,
        score_col=score_col,
        threshold=threshold,
        window_start=window_start,
        window_end=window_end,
    )

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

    # Current ("vertical") gap snapshot: today's best closed minus best open.
    # Kept alongside the windowed average for back-compatibility.
    best_open_score = df_open[score_col].max() if len(df_open) > 0 else 0
    best_closed_score = df_closed[score_col].max() if len(df_closed) > 0 else 0
    vertical_gap = best_closed_score - best_open_score

    # Estimate current gap using unmatched models
    current_gap_estimate = estimate_current_gap(
        gaps, matched_gaps_months,
        use_survival_analysis=use_survival_analysis,
        prior_from_first_match=prior_from_first_match,
    )

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
        "current_gap_estimate": current_gap_estimate,
    }

# Chinese tech companies/organizations
CHINA_ORGANIZATIONS = {
    "DeepSeek",
    "Alibaba",
    "Baichuan",
    "01.AI",
    "Moonshot",
    "Tsinghua",
    "Peking University",
    "ByteDance",
    "Tencent",
    "Huawei",
    "SenseTime",
    "iFlytek",
    "Zhipu AI",
}

# US tech companies/organizations
US_ORGANIZATIONS = {
    "OpenAI",
    "Anthropic",
    "Google",
    "Google DeepMind",
    "Meta",
    "Meta AI",
    "Microsoft",
    "Microsoft Research",
    "xAI",
    "NVIDIA",
    "Databricks",
    "MosaicML",
    "Salesforce",
    "Cerebras",
    "Hugging Face",
}


def is_china_org(org: str) -> bool:
    """Check if an organization is Chinese."""
    if not isinstance(org, str):
        return False
    org_lower = org.lower()
    for china_org in CHINA_ORGANIZATIONS:
        if china_org.lower() in org_lower:
            return True
    return False


def is_us_org(org: str) -> bool:
    """Check if an organization is US-based."""
    if not isinstance(org, str):
        return False
    org_lower = org.lower()
    for us_org in US_ORGANIZATIONS:
        if us_org.lower() in org_lower:
            return True
    return False


def process_data() -> dict[str, Any]:
    """Process ECI data and calculate gaps."""
    df = fetch_eci_data()
    df["Open"] = df["Model accessibility"].str.contains("Open", na=False)

    df["is_china"] = df["Organization"].apply(is_china_org)
    df["is_us"] = df["Organization"].apply(is_us_org)

    df_open = df[df["Open"]].copy()
    df_closed = df[~df["Open"]].copy()

    df_open["group_rank"] = get_rank(df_open, sort_col="date", val_col="eci")
    df_closed["group_rank"] = get_rank(df_closed, sort_col="date", val_col="eci")

    df_combined = pd.concat([df_open, df_closed]).sort_values("date")
    df_frontier = df_combined[df_combined["group_rank"] <= 1].copy()

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
            "is_china": bool(row.get("is_china", False)),
        })

    trend_models = []
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
            "is_china": bool(row.get("is_china", False)),
        })

    gaps = calculate_horizontal_gaps(df_frontier, score_col="eci", threshold=ECI_MATCH_THRESHOLD, model_col="Model")
    # Rename score fields to eci for backward compatibility
    for gap in gaps:
        gap["closed_eci"] = gap.pop("closed_score")
        gap["open_eci"] = gap.pop("open_score")

    stats = calculate_statistics(df_frontier, gaps, score_col="eci")
    trends = calculate_trends(df_all_valid, use_apr_2024_split=True)  # ECI uses April 2024 split

    # Calculate historical gaps for the timeline chart
    historical_gaps = calculate_historical_gaps(df_frontier, score_col="eci", threshold=ECI_MATCH_THRESHOLD, model_col="Model")
    # Rename score fields to eci for backward compatibility
    for hg in historical_gaps:
        hg["reference_eci"] = hg.pop("reference_score")
        hg["open_frontier_eci"] = hg.pop("open_frontier_score")

    # Also calculate gaps using China vs US framing
    # Filter to only China and US models
    df_china_us = df[df["is_china"] | df["is_us"]].copy()
    df_china_us["Open"] = df_china_us["is_china"]  # China = "Open" (catching up), US = "Closed" (leading)

    df_china_models = df_china_us[df_china_us["Open"]].copy()
    df_us_models = df_china_us[~df_china_us["Open"]].copy()

    if len(df_china_models) > 0 and len(df_us_models) > 0:
        df_china_models["group_rank"] = get_rank(df_china_models, sort_col="date", val_col="eci")
        df_us_models["group_rank"] = get_rank(df_us_models, sort_col="date", val_col="eci")
        df_china_us_combined = pd.concat([df_china_models, df_us_models]).sort_values("date")
        df_china_us_frontier = df_china_us_combined[df_china_us_combined["group_rank"] <= 1].copy()

        china_gaps = calculate_horizontal_gaps(df_china_us_frontier, score_col="eci", threshold=ECI_MATCH_THRESHOLD, model_col="Model")
        for gap in china_gaps:
            gap["closed_eci"] = gap.pop("closed_score")
            gap["open_eci"] = gap.pop("open_score")

        china_stats = calculate_statistics(df_china_us_frontier, china_gaps, score_col="eci")

        china_historical = calculate_historical_gaps(df_china_us_frontier, score_col="eci", threshold=ECI_MATCH_THRESHOLD, model_col="Model")
        for hg in china_historical:
            hg["reference_eci"] = hg.pop("reference_score")
            hg["open_frontier_eci"] = hg.pop("open_frontier_score")
    else:
        china_gaps = []
        china_stats = {}
        china_historical = []

    return {
        "models": models,
        "trend_models": trend_models,
        "gaps": gaps,
        "statistics": stats,
        "trends": trends,
        "historical_gaps": historical_gaps,
        "china_framing": {
            "gaps": china_gaps,
            "statistics": china_stats,
            "historical_gaps": china_historical,
        },
        "last_updated": datetime.now().isoformat(),
    }

def process_benchmark_data(benchmark_id: str, benchmark_data: dict) -> Optional[dict]:
    """
    Process benchmark data from Airtable into the same format as ECI.

    Args:
        benchmark_id: The benchmark identifier (e.g., "gpqa_diamond")
        benchmark_data: Data from BenchmarkDataFetcher.fetch_benchmark_data()

    Returns:
        Processed benchmark data with gaps, statistics, etc.
    """
    if not benchmark_data or not benchmark_data.get("models"):
        logger.warning(f"No data for benchmark {benchmark_id}")
        return None

    models = benchmark_data["models"]
    metadata = benchmark_data["metadata"]
    threshold = metadata.get("threshold", 1.0)

    logger.info(f"Processing {metadata['name']} with {len(models)} models...")

    # Convert to DataFrame for processing
    df = pd.DataFrame(models)
    df["date"] = pd.to_datetime(df["date"])
    df["Open"] = df["is_open"]

    # Rename score column to match ECI pipeline expectations
    df["score"] = df["score"]
    df["score_std"] = df.get("score_std", pd.NA)

    # Get ranks for each group
    df_open = df[df["Open"]].copy()
    df_closed = df[~df["Open"]].copy()

    if len(df_open) == 0 or len(df_closed) == 0:
        logger.warning(f"Insufficient data for {benchmark_id}: {len(df_open)} open, {len(df_closed)} closed")
        return None

    df_open["group_rank"] = get_rank(df_open, sort_col="date", val_col="score")
    df_closed["group_rank"] = get_rank(df_closed, sort_col="date", val_col="score")

    df_combined = pd.concat([df_open, df_closed]).sort_values("date")
    df_frontier = df_combined[df_combined["group_rank"] <= 1].copy()

    # Build model list for output
    output_models = []
    for _, row in df_frontier.iterrows():
        output_models.append({
            "model": row.get("model", "Unknown"),
            "display_name": row.get("display_name", row.get("model", "Unknown")),
            "score": float(row["score"]) if pd.notna(row["score"]) else None,
            "score_std": float(row["score_std"]) if pd.notna(row.get("score_std")) else None,
            "date": row["date"].isoformat() if pd.notna(row["date"]) else None,
            "organization": row.get("organization", "Unknown"),
            "is_open": bool(row["Open"]),
            "is_china": bool(row.get("is_china", False)),
        })

    # Calculate gaps
    gaps = calculate_horizontal_gaps(
        df_frontier,
        score_col="score",
        threshold=threshold,
        model_col="model"
    )

    # Calculate statistics
    stats = calculate_statistics(df_frontier, gaps, score_col="score", threshold=threshold)

    # Calculate historical gaps
    historical_gaps = calculate_historical_gaps(
        df_frontier,
        score_col="score",
        threshold=threshold,
        model_col="model"
    )

    # Build trend models (all models, not just frontier)
    trend_models = []
    for _, row in df_combined.iterrows():
        if pd.isna(row["score"]) or pd.isna(row["date"]):
            continue
        trend_models.append({
            "model": row.get("model", "Unknown"),
            "display_name": row.get("display_name", row.get("model", "Unknown")),
            "score": float(row["score"]),
            "score_std": float(row["score_std"]) if pd.notna(row.get("score_std")) else None,
            "date": row["date"].isoformat(),
            "organization": row.get("organization", "Unknown"),
            "is_open": bool(row["Open"]),
            "is_china": bool(row.get("is_china", False)),
        })

    # Calculate trends
    trends = calculate_trends(df_combined.rename(columns={"score": "eci"}))

    return {
        "metadata": metadata,
        "models": output_models,
        "trend_models": trend_models,
        "gaps": gaps,
        "statistics": stats,
        "trends": trends,
        "historical_gaps": historical_gaps,
    }


def process_metr_data(metr_raw: dict) -> Optional[dict]:
    """
    Process METR Time Horizon data into the app's benchmark format.

    METR data uses p50_horizon_minutes as the primary score (how long a task
    the model can complete at 50% success rate). For gap analysis, we use
    this metric since it directly measures capability in a meaningful unit.

    For matching: an open model "matches" a closed model if its p50 horizon
    is at least half of the closed model's (ratio-based threshold).
    """
    models_raw = metr_raw.get("models", [])
    metadata = metr_raw.get("metadata", {})

    if not models_raw:
        logger.warning("No METR models to process")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(models_raw)
    df["date"] = pd.to_datetime(df["date"])
    df["Open"] = df["is_open"]

    # Use p50_horizon_minutes as the primary score
    # Filter out models without p50 data
    df["score"] = df["p50_horizon_minutes"]
    df = df.dropna(subset=["score", "date"]).copy()

    if len(df) == 0:
        logger.warning("No METR models with valid p50 horizon data")
        return None

    df_open = df[df["Open"]].copy()
    df_closed = df[~df["Open"]].copy()

    if len(df_open) == 0 or len(df_closed) == 0:
        logger.warning(f"METR: insufficient data ({len(df_open)} open, {len(df_closed)} closed)")
        return None

    # Calculate ranks
    df_open["group_rank"] = get_rank(df_open, sort_col="date", val_col="score")
    df_closed["group_rank"] = get_rank(df_closed, sort_col="date", val_col="score")

    df_combined = pd.concat([df_open, df_closed]).sort_values("date")
    df_frontier = df_combined[df_combined["group_rank"] <= 1].copy()

    # Build frontier model list
    output_models = []
    for _, row in df_frontier.iterrows():
        output_models.append({
            "model": row["model"],
            "display_name": row["display_name"],
            "score": float(row["score"]),
            "score_std": None,  # METR uses CI instead of std
            "p50_ci_low": float(row["p50_ci_low"]) if pd.notna(row.get("p50_ci_low")) else None,
            "p50_ci_high": float(row["p50_ci_high"]) if pd.notna(row.get("p50_ci_high")) else None,
            "average_score": float(row["average_score"]),
            "p80_horizon_minutes": float(row["p80_horizon_minutes"]) if pd.notna(row.get("p80_horizon_minutes")) else None,
            "date": row["date"].isoformat() if pd.notna(row["date"]) else None,
            "organization": row.get("organization", "Unknown"),
            "is_open": bool(row["Open"]),
            "is_china": bool(row.get("is_china", False)),
            "is_sota": bool(row.get("is_sota", False)),
            "source_version": row.get("source_version", ""),
        })

    # Build trend models (all models, not just frontier)
    trend_models = []
    for _, row in df_combined.iterrows():
        if pd.isna(row["score"]) or pd.isna(row["date"]):
            continue
        trend_models.append({
            "model": row["model"],
            "display_name": row["display_name"],
            "score": float(row["score"]),
            "score_std": None,
            "average_score": float(row["average_score"]),
            "p50_ci_low": float(row["p50_ci_low"]) if pd.notna(row.get("p50_ci_low")) else None,
            "p50_ci_high": float(row["p50_ci_high"]) if pd.notna(row.get("p50_ci_high")) else None,
            "p80_horizon_minutes": float(row["p80_horizon_minutes"]) if pd.notna(row.get("p80_horizon_minutes")) else None,
            "date": row["date"].isoformat(),
            "organization": row.get("organization", "Unknown"),
            "is_open": bool(row["Open"]),
            "is_china": bool(row.get("is_china", False)),
            "is_sota": bool(row.get("is_sota", False)),
            "source_version": row.get("source_version", ""),
        })

    # METR uses exact matching (open model must meet or exceed closed model's horizon)
    gaps = calculate_horizontal_gaps(
        df_frontier,
        score_col="score",
        threshold=0,
        model_col="model"
    )

    stats = calculate_statistics(df_frontier, gaps, score_col="score", use_survival_analysis=False, threshold=0)

    historical_gaps = calculate_historical_gaps(
        df_frontier,
        score_col="score",
        threshold=0,
        model_col="model"
    )

    # Calculate trends using log scale for horizon (exponential growth)
    trends = calculate_trends(
        df_combined.rename(columns={"score": "eci"}),
        use_apr_2024_split=False
    )

    # China framing
    df_china_us = df[df["is_china"] | df["is_us"]].copy()
    df_china_us["Open"] = df_china_us["is_china"]

    china_framing = {"gaps": [], "statistics": {}, "historical_gaps": []}
    df_china_models = df_china_us[df_china_us["Open"]].copy()
    df_us_models = df_china_us[~df_china_us["Open"]].copy()

    if len(df_china_models) > 0 and len(df_us_models) > 0:
        df_china_models["group_rank"] = get_rank(df_china_models, sort_col="date", val_col="score")
        df_us_models["group_rank"] = get_rank(df_us_models, sort_col="date", val_col="score")
        df_cu_combined = pd.concat([df_china_models, df_us_models]).sort_values("date")
        df_cu_frontier = df_cu_combined[df_cu_combined["group_rank"] <= 1].copy()

        china_gaps = calculate_horizontal_gaps(df_cu_frontier, score_col="score", threshold=0, model_col="model")
        china_stats = calculate_statistics(df_cu_frontier, china_gaps, score_col="score", use_survival_analysis=False, threshold=0)
        china_historical = calculate_historical_gaps(df_cu_frontier, score_col="score", threshold=0, model_col="model")
        china_framing = {
            "gaps": china_gaps,
            "statistics": china_stats,
            "historical_gaps": china_historical,
        }

    return {
        "metadata": metadata,
        "models": output_models,
        "trend_models": trend_models,
        "gaps": gaps,
        "statistics": stats,
        "trends": trends,
        "historical_gaps": historical_gaps,
        "china_framing": china_framing,
    }


def process_all_benchmarks() -> dict:
    """Process all available benchmarks including ECI, METR, and Airtable benchmarks."""
    benchmarks = {}

    # Process ECI (always available)
    logger.info("Processing ECI benchmark...")
    eci_data = process_data()

    # Convert ECI data to benchmark format (for consistency)
    benchmarks["eci"] = {
        "metadata": {
            "id": "eci",
            "name": "Epoch Capabilities Index (ECI)",
            "description": "Comprehensive AI capability index from Epoch AI",
            "unit": "ECI Score",
            "threshold": ECI_MATCH_THRESHOLD,
            "scale": 1,
        },
        "models": eci_data["models"],
        "trend_models": eci_data["trend_models"],
        "gaps": eci_data["gaps"],
        "statistics": eci_data["statistics"],
        "trends": eci_data["trends"],
        "historical_gaps": eci_data["historical_gaps"],
        "china_framing": eci_data["china_framing"],
    }

    # Process METR Time Horizon benchmark
    if METR_FETCHER_AVAILABLE:
        logger.info("Processing METR Time Horizon benchmark...")
        try:
            metr_raw = fetch_metr_data()
            if metr_raw:
                metr_processed = process_metr_data(metr_raw)
                if metr_processed:
                    benchmarks["metr_time_horizon"] = metr_processed
                    logger.info(f"  METR: {len(metr_processed['models'])} frontier models, "
                                f"{len(metr_processed['trend_models'])} total")
                else:
                    logger.warning("  METR processing returned no data")
            else:
                logger.warning("  METR fetch returned no data")
        except Exception as e:
            logger.error(f"  Error processing METR data: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning("METR fetcher not available")

    # Process additional benchmarks from CSV exports (preferred - no credentials needed)
    if CSV_FETCHER_AVAILABLE:
        logger.info("Fetching additional benchmarks from Epoch AI CSV exports...")
        try:
            fetcher = CSVBenchmarkFetcher()

            # Process each configured benchmark
            for benchmark_id in BENCHMARK_CSV_CONFIG.keys():
                try:
                    logger.info(f"Fetching {benchmark_id}...")
                    raw_data = fetcher.fetch_benchmark_data(benchmark_id)
                    if raw_data:
                        processed = process_benchmark_data(benchmark_id, raw_data)
                        if processed:
                            benchmarks[benchmark_id] = processed
                            logger.info(f"  Successfully processed {benchmark_id}")
                        else:
                            logger.warning(f"  Skipped {benchmark_id} (insufficient data)")
                except Exception as e:
                    logger.error(f"  Error processing {benchmark_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch CSV benchmarks: {e}")
            import traceback
            traceback.print_exc()

    # Fallback to Airtable fetcher if CSV fetcher unavailable or for additional benchmarks
    elif AIRTABLE_FETCHER_AVAILABLE:
        logger.info("Fetching additional benchmarks from Epoch AI Airtable...")
        try:
            fetcher = BenchmarkDataFetcher()

            # Process each configured benchmark
            for benchmark_id in BENCHMARK_CONFIG.keys():
                if benchmark_id in benchmarks:
                    continue  # Skip if already fetched from CSV
                try:
                    logger.info(f"Fetching {benchmark_id}...")
                    raw_data = fetcher.fetch_benchmark_data(benchmark_id)
                    if raw_data:
                        processed = process_benchmark_data(benchmark_id, raw_data)
                        if processed:
                            benchmarks[benchmark_id] = processed
                            logger.info(f"  Successfully processed {benchmark_id}")
                        else:
                            logger.warning(f"  Skipped {benchmark_id} (insufficient data)")
                except Exception as e:
                    logger.error(f"  Error processing {benchmark_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize Airtable benchmark fetcher: {e}")
    else:
        logger.warning("No benchmark fetcher available - only ECI and METR will be processed")

    return {
        "benchmarks": benchmarks,
        "default_benchmark": "eci",
        "last_updated": datetime.now().isoformat(),
    }


def main():
    try:
        # Process all benchmarks
        data = process_all_benchmarks()

        # Preserve benchmarks from existing data.json that we couldn't regenerate
        # (e.g., Airtable benchmarks when credentials aren't available locally)
        try:
            with open("data.json", "r") as f:
                existing_data = json.load(f)
            existing_benchmarks = existing_data.get("benchmarks", {})
            for bid, bdata in existing_benchmarks.items():
                if bid not in data["benchmarks"]:
                    data["benchmarks"][bid] = bdata
                    logger.info(f"  Preserved existing benchmark: {bid}")
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # Write the new multi-benchmark format
        with open("data.json", "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully wrote data.json with {len(data['benchmarks'])} benchmarks")

        # List processed benchmarks
        for bid, bdata in data["benchmarks"].items():
            name = bdata.get("metadata", {}).get("name", bid)
            model_count = len(bdata.get("models", []))
            logger.info(f"  - {name}: {model_count} frontier models")

    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
