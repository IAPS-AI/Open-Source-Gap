
import json
import logging
import math
import sys
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress, norm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data source
ECI_SCORES_URL = "https://epoch.ai/data/eci_scores.csv"

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

def fetch_eci_data() -> pd.DataFrame:
    """Fetch ECI scores from Epoch AI."""
    logger.info("Fetching data from Epoch AI...")
    try:
        df = pd.read_csv(ECI_SCORES_URL)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
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

def calculate_horizontal_gaps(df: pd.DataFrame) -> list[dict]:
    """Calculate horizontal gaps."""
    df_open = df[df["Open"]].sort_values("date")
    df_closed = df[~df["Open"]].sort_values("date")

    gaps = []

    for _, closed_row in df_closed.iterrows():
        closed_eci = closed_row["eci"]
        closed_date = closed_row["date"]

        if pd.isna(closed_eci) or pd.isna(closed_date):
            continue

        matching_open = None
        match_type = None
        for _, open_row in df_open.iterrows():
            if pd.isna(open_row["eci"]) or pd.isna(open_row["date"]):
                continue

            if open_row["date"] <= closed_date:
                continue

            if open_row["eci"] >= closed_eci:
                matching_open = open_row
                match_type = "exact"
                break

        if matching_open is not None:
            gap_days = (matching_open["date"] - closed_date).days
            gap_months = gap_days / 30.5

            gaps.append({
                "closed_model": closed_row.get("Model", "Unknown"),
                "closed_date": closed_date.isoformat(),
                "closed_eci": float(closed_eci),
                "open_model": matching_open.get("Model", "Unknown"),
                "open_date": matching_open["date"].isoformat(),
                "open_eci": float(matching_open["eci"]),
                "gap_months": round(gap_months, 1),
                "matched": True,
                "match_type": match_type,
            })
        else:
            now = datetime.now()
            gap_days = (now - closed_date.to_pydatetime().replace(tzinfo=None)).days
            gap_months = gap_days / 30.5

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
    """Calculate trends for models before and after Apr 2024."""
    trends = {}
    cutoff = pd.Timestamp("2024-04-01")
    
    def get_stats(sub_df, name):
        if len(sub_df) < 2:
            return None
            
        dates_ordinal = sub_df["date"].map(datetime.toordinal).values
        ecis = sub_df["eci"].values
        
        lin_slope, lin_intercept, _, _, _ = linregress(dates_ordinal, ecis)
        yearly_absolute_growth = lin_slope * 365.25
        
        start_date_ord = dates_ordinal.min()
        end_date_ord = dates_ordinal.max()
        
        lin_start_eci = lin_slope * start_date_ord + lin_intercept
        lin_end_eci = lin_slope * end_date_ord + lin_intercept
        
        valid_indices = ecis > 0
        if not np.any(valid_indices):
            return None
            
        log_ecis = np.log(ecis[valid_indices])
        log_dates = dates_ordinal[valid_indices]
        
        exp_slope, _, _, _, _ = linregress(log_dates, log_ecis)
        k_annual = exp_slope * 365.25
        
        pct_growth = (np.exp(k_annual) - 1) * 100
        multiples_per_year = np.exp(k_annual)
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

    pre_2024 = df[df["date"] < cutoff]
    trends["pre_apr_2024"] = get_stats(pre_2024, "Pre-Apr 2024")
    
    post_2024 = df[df["date"] >= cutoff]
    trends["post_apr_2024"] = get_stats(post_2024, "Post-Apr 2024")
    
    return trends

def calculate_statistics(df: pd.DataFrame, gaps: list[dict]) -> dict:
    """Calculate summary statistics."""
    df_open = df[df["Open"]].copy()
    df_closed = df[~df["Open"]].copy()
    
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
    end_eci = max(df_open["eci"].max(), df_closed["eci"].max())
    
    horizontal_gaps = []
    df_open_possible = df_open.sort_values("date").copy()
    
    for cur_eci in np.linspace(start_eci, end_eci, 100):
        closed_candidates = df_closed[df_closed["eci"] >= cur_eci].sort_values("date")
        if len(closed_candidates) == 0:
            continue
        cur_closed_model = closed_candidates.iloc[0]
        
        cur_open_model = None
        for _, row in df_open_possible.iterrows():
            if pd.isna(row["eci"]) or pd.isna(row["date"]):
                continue
            if row["eci"] >= cur_eci:
                cur_open_model = row
                gap = (cur_open_model["date"] - cur_closed_model["date"]).days / 30.5
                horizontal_gaps.append(gap)
                break
        
        if cur_open_model is None:
            now = datetime.now()
            gap = (now - cur_closed_model["date"].to_pydatetime().replace(tzinfo=None)).days / 30.5
            horizontal_gaps.append(gap)
    
    if horizontal_gaps:
        avg_gap = np.mean(horizontal_gaps)
        std_gap = np.std(horizontal_gaps)
        ci_low = np.percentile(horizontal_gaps, 5)
        ci_high = np.percentile(horizontal_gaps, 95)
    else:
        avg_gap = std_gap = ci_low = ci_high = 0

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

def process_data() -> dict[str, Any]:
    """Process ECI data and calculate gaps."""
    df = fetch_eci_data()
    df["Open"] = df["Model accessibility"].str.contains("Open", na=False)

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
        })

    gaps = calculate_horizontal_gaps(df_frontier)
    stats = calculate_statistics(df_frontier, gaps)
    trends = calculate_trends(df_all_valid)

    return {
        "models": models,
        "trend_models": trend_models,
        "gaps": gaps,
        "statistics": stats,
        "trends": trends,
        "last_updated": datetime.now().isoformat(),
    }

def main():
    try:
        data = process_data()
        with open("data.json", "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Successfully wrote data.json")
    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
