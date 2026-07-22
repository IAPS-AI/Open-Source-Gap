"""Release-lag bracket and trend-regression gap estimators.

Motivated by the gap methodology in "Have Chinese AI Models Caught Up to the
US Frontier?" (Lisan al Gaib, July 19, 2026; PDF in the repo root). Observable
progress is discrete -- models launch every few weeks to months -- so a single
"time since the leader hit the laggard's level" number silently picks one of
three estimators. This module makes all three explicit for the open-vs-closed
(and China-vs-US) framing:

1. ``current_lag_bracket`` -- the instantaneous adjacent-model bracket:
   * OVER-estimate: time since the most recent leader-frontier model at or
     below the laggard's current best score (the leader demonstrably reached
     the level no later than the *next* release, so measuring from this one
     overstates the lag);
   * UNDER-estimate: time since the earliest leader-frontier model strictly
     above the laggard's current best score;
   * CENTRAL estimate: interpolate between those two adjacent leader models
     (in transform space, see below) for the date the leader "crossed" the
     laggard's current level.

2. ``compute_trend_gap`` -- fit one regression line per group to the frontier
   series and compare the lines instead of individual releases:
   * backward-looking trend gap: horizontal offset between the two lines at
     the evaluation date;
   * gap velocity: months of gap gained/lost per year (from the slope ratio);
   * forward-looking gap: when the laggard's trend line is projected to reach
     the leader's current best OBSERVED score. This is a forecast -- it
     assumes both trends hold -- and is reported separately from the
     empirical backward-looking numbers.
   Bootstrap (resampling models within each group) supplies 90% CIs.

Scores are fitted/interpolated in a transform space appropriate to the
metric, following the article's treatment of bounded indexes:

* ``linear`` -- unbounded indexes (ECI);
* ``logit``  -- percentage scores bounded at 0-100 (GPQA Diamond, MATH L5,
  ...). Raw-space fits compress progress near the floor and ceiling; the
  logit undoes that compression;
* ``log``    -- positive quantities with exponential growth (METR horizon
  minutes).

Both estimators expect the per-group frontier (running-max records per
group), with a boolean ``Open`` column where True marks the LAGGARD group
(open-weight models, or Chinese models in the China-vs-US framing) -- the
same convention as ``scripts/update_data.py``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 365.25 / 12  # matches scripts/update_data.py

# Percentage scores are clipped into [LOGIT_CLIP, 100 - LOGIT_CLIP] before the
# logit so observed 0s / 100s stay finite.
LOGIT_CLIP = 0.1

# Bootstrap defaults: 90% percentile CIs from resampling models within each
# group; draws with a degenerate refit (single date, non-positive leader
# slope) are discarded, and CIs are suppressed when too few draws survive.
BOOT_N = 2000
BOOT_SEED = 0
BOOT_MIN_VALID = 100
CI_QUANTILES = (0.05, 0.95)
# A percentile bootstrap over fewer records than this has no meaningful 90%
# coverage (with n=3 there are only ~10 distinct resamples), so CIs are
# suppressed entirely for smaller groups; point estimates are still reported.
MIN_N_FOR_CI = 8
# Forward CIs are only published when at least this share of otherwise-valid
# bootstrap draws produced a crossing -- otherwise the interval is heavily
# conditional on "catch-up happens at all" and understates uncertainty.
FORWARD_CI_MIN_SHARE = 0.9
# Projected crossings further out than this are trend-fit noise, not
# forecasts; report None ("no projected catch-up at current trends").
MAX_FORWARD_MONTHS = 240.0
_MAX_ORDINAL = datetime.max.toordinal()


def transform_scores(values: np.ndarray, transform: str) -> np.ndarray:
    """Map raw scores into fit space. Values must be float ndarray."""
    if transform == "linear":
        return values.astype(float)
    if transform == "logit":
        p = np.clip(values.astype(float), LOGIT_CLIP, 100.0 - LOGIT_CLIP)
        return np.log(p / (100.0 - p))
    if transform == "log":
        return np.log(values.astype(float))
    raise ValueError(f"Unknown transform: {transform}")


def _valid_for_transform(values: pd.Series, transform: str) -> pd.Series:
    """Rows usable under the transform (log requires positive scores)."""
    if transform == "log":
        return values > 0
    return values.notna()


def _prepare_groups(
    df: pd.DataFrame,
    score_col: str,
    transform: str,
) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    if df is None or len(df) == 0 or "Open" not in df.columns:
        return None
    d = df.dropna(subset=["date", score_col]).copy()
    d = d[_valid_for_transform(d[score_col], transform)]
    if d.empty:
        return None
    d = d.sort_values("date", kind="mergesort")
    laggard = d[d["Open"]]
    leader = d[~d["Open"]]
    if laggard.empty or leader.empty:
        return None
    return laggard, leader


def _running_max_records(
    rows: pd.DataFrame, score_col: str, model_col: str
) -> list[dict]:
    """Date-ordered records that set a new group record (running max)."""
    out: list[dict] = []
    run = -np.inf
    for _, r in rows.iterrows():
        s = float(r[score_col])
        if s > run:
            run = s
            out.append({
                "model": r.get(model_col, r.get("model")),
                "date": r["date"],
                "score": s,
            })
    return out


def current_lag_bracket(
    df: pd.DataFrame,
    score_col: str = "score",
    model_col: str = "model",
    transform: str = "linear",
    as_of: Optional[Any] = None,
) -> Optional[dict]:
    """Adjacent-model lag bracket for the laggard's CURRENT frontier level.

    Point-estimate based by design: the bracket quantifies the uncertainty
    coming from discrete release timing, which is a separate axis from score
    measurement uncertainty (the CI/bootstrap catch-up criterion in
    ``update_data.py`` handles the latter). Returns None when either group is
    empty; sets ``laggard_leads`` when the laggard's best score tops every
    leader model, and ``censored`` when the leader was already above the
    laggard's level at the start of observed history (the over-estimate and
    central estimate are then unknown, not zero).

    ``as_of`` defaults to the latest release date in the supplied data --
    the same convention as :func:`compute_trend_gap` -- so the two
    estimators published side by side answer the question at the same date,
    stale benchmarks are not presented as current, and regenerating with
    unchanged data produces unchanged output.
    """
    groups = _prepare_groups(df, score_col, transform)
    if groups is None:
        return None
    laggard, leader = groups

    as_of_ts = (
        pd.Timestamp(as_of)
        if as_of is not None
        else max(laggard["date"].max(), leader["date"].max())
    )

    lag_best_row = laggard.loc[laggard[score_col].idxmax()]
    lag_best = float(lag_best_row[score_col])

    sota = _running_max_records(leader, score_col, model_col)
    if not sota:
        return None

    below = None
    above = None
    for rec in sota:  # ascending in both date and score
        if rec["score"] <= lag_best:
            below = rec
        elif above is None:
            above = rec
            break

    def _months_since(ts: pd.Timestamp) -> float:
        return (as_of_ts - ts).days / DAYS_PER_MONTH

    result: dict[str, Any] = {
        "as_of": as_of_ts.isoformat(),
        "transform": transform,
        "laggard_model": lag_best_row.get(model_col, lag_best_row.get("model")),
        "laggard_score": round(lag_best, 2),
        "laggard_leads": above is None,
        "censored": below is None and above is not None,
        "below_model": None,
        "above_model": None,
        "under_months": None,
        "over_months": None,
        "central_months": None,
    }

    if above is None:
        # Laggard tops every leader release: no lag at the frontier.
        result["under_months"] = 0.0
        result["over_months"] = 0.0
        result["central_months"] = 0.0
        if below is not None:
            result["below_model"] = {
                "model": below["model"],
                "date": below["date"].isoformat(),
                "score": round(below["score"], 2),
            }
        return result

    result["above_model"] = {
        "model": above["model"],
        "date": above["date"].isoformat(),
        "score": round(above["score"], 2),
    }
    result["under_months"] = round(_months_since(above["date"]), 2)

    if below is None:
        # Leader was above this level before observed history began: the lag
        # is at least the under-estimate, but the over/central are unknown.
        return result

    result["below_model"] = {
        "model": below["model"],
        "date": below["date"].isoformat(),
        "score": round(below["score"], 2),
    }
    result["over_months"] = round(_months_since(below["date"]), 2)

    # Central estimate: interpolate the leader's crossing of the laggard's
    # level between the two adjacent leader records, in transform space.
    t_below, t_above, t_lag = transform_scores(
        np.array([below["score"], above["score"], lag_best]), transform
    )
    span = t_above - t_below
    frac = 0.0 if span <= 0 else float(np.clip((t_lag - t_below) / span, 0.0, 1.0))
    cross_days = (above["date"] - below["date"]).days * frac
    cross_date = below["date"] + pd.Timedelta(days=cross_days)
    result["central_months"] = round(_months_since(cross_date), 2)
    result["crossing_date"] = cross_date.isoformat()
    return result


def _fit_line(dates_ord: np.ndarray, y: np.ndarray) -> Optional[tuple[float, float]]:
    """OLS slope/intercept of y ~ date-ordinal; None if dates are degenerate."""
    if len(dates_ord) < 2 or len(set(dates_ord.tolist())) < 2:
        return None
    # Center dates before fitting: raw ordinals (~739000) are poorly
    # conditioned for polyfit.
    x0 = float(np.mean(dates_ord))
    slope, c = np.polyfit(dates_ord - x0, y, 1)
    return float(slope), float(c - slope * x0)


def _trend_metrics(
    m_lag: float,
    b_lag: float,
    m_lead: float,
    b_lead: float,
    t_eval: float,
    leader_best_t: float,
) -> Optional[dict]:
    """Backward gap / velocity / forward gap from two fitted lines.

    ``t_eval`` and outputs are in day units; leader_best_t is the leader's
    current best score in transform space. Requires a positive leader slope
    (otherwise the leader line cannot be inverted for a horizontal offset).
    """
    if m_lead <= 0:
        return None
    y_lag_now = m_lag * t_eval + b_lag
    t_lead_at_level = (y_lag_now - b_lead) / m_lead
    backward_months = (t_eval - t_lead_at_level) / DAYS_PER_MONTH

    # d(gap)/dt = 1 - m_lag/m_lead (days of gap per day); scale to months/year.
    velocity_months_per_year = (1.0 - m_lag / m_lead) * 12.0

    forward_months = None
    crossing_ord = None
    if m_lag > 0:
        crossing_ord = (leader_best_t - b_lag) / m_lag
        forward_months = (crossing_ord - t_eval) / DAYS_PER_MONTH
        # A near-flat laggard slope projects the crossing centuries out
        # (overflowing datetime's year-9999 ordinal range); such projections
        # are fit noise, not forecasts -- report "no projected catch-up".
        if (
            forward_months > MAX_FORWARD_MONTHS
            or crossing_ord < 1
            or crossing_ord > _MAX_ORDINAL
        ):
            forward_months = None
            crossing_ord = None

    return {
        "backward_months": backward_months,
        "velocity_months_per_year": velocity_months_per_year,
        "forward_months": forward_months,
        "crossing_ord": crossing_ord,
    }


def compute_trend_gap(
    df: pd.DataFrame,
    score_col: str = "score",
    model_col: str = "model",
    transform: str = "linear",
    min_points: int = 3,
    n_boot: int = BOOT_N,
    seed: int = BOOT_SEED,
    as_of: Optional[Any] = None,
) -> Optional[dict]:
    """Regression-trendline gap between the leader and laggard frontiers.

    Fits one OLS line per group (transformed score ~ release date) over the
    supplied frontier records and reports the backward-looking horizontal
    offset, the gap velocity, and the forward-looking catch-up projection
    with bootstrap 90% CIs. Returns None when either group has fewer than
    ``min_points`` usable records or the leader trend is flat/decreasing.
    CIs are suppressed (None) when either group has fewer than
    ``MIN_N_FOR_CI`` records, and the forward CI additionally requires at
    least ``FORWARD_CI_MIN_SHARE`` of valid draws to produce a crossing.
    Projections further out than ``MAX_FORWARD_MONTHS`` are reported as None
    (no meaningful catch-up at current trends).

    The trend estimate reflects the broad multi-release trajectory; it will
    lag genuine breaks in trend in either direction (the article counts only
    ~2 such breaks in 5 years), which is why the instantaneous
    ``current_lag_bracket`` is published alongside it.
    """
    groups = _prepare_groups(df, score_col, transform)
    if groups is None:
        return None
    laggard, leader = groups
    if len(laggard) < min_points or len(leader) < min_points:
        return None

    lag_ord = laggard["date"].map(datetime.toordinal).to_numpy(dtype=float)
    lead_ord = leader["date"].map(datetime.toordinal).to_numpy(dtype=float)
    lag_y = transform_scores(laggard[score_col].to_numpy(dtype=float), transform)
    lead_y = transform_scores(leader[score_col].to_numpy(dtype=float), transform)

    fit_lag = _fit_line(lag_ord, lag_y)
    fit_lead = _fit_line(lead_ord, lead_y)
    if fit_lag is None or fit_lead is None:
        return None
    m_lag, b_lag = fit_lag
    m_lead, b_lead = fit_lead

    as_of_ts = (
        pd.Timestamp(as_of)
        if as_of is not None
        else max(laggard["date"].max(), leader["date"].max())
    )
    t_eval = float(pd.Timestamp(as_of_ts).to_pydatetime().toordinal())

    lead_best_idx = leader[score_col].idxmax()
    leader_best_score = float(leader.loc[lead_best_idx, score_col])
    leader_best_t = float(transform_scores(np.array([leader_best_score]), transform)[0])

    point = _trend_metrics(m_lag, b_lag, m_lead, b_lead, t_eval, leader_best_t)
    if point is None:
        return None

    # Bootstrap over models within each group. CIs have no meaningful 90%
    # coverage for very small groups, so skip the bootstrap entirely then.
    rng = np.random.default_rng(seed)
    boot_backward: list[float] = []
    boot_velocity: list[float] = []
    boot_forward: list[float] = []
    boot_crossing: list[float] = []
    boot_attempts = 0
    do_boot = len(lag_ord) >= MIN_N_FOR_CI and len(lead_ord) >= MIN_N_FOR_CI
    for _ in range(n_boot if do_boot else 0):
        li = rng.integers(0, len(lag_ord), len(lag_ord))
        ci = rng.integers(0, len(lead_ord), len(lead_ord))
        f_lag = _fit_line(lag_ord[li], lag_y[li])
        f_lead = _fit_line(lead_ord[ci], lead_y[ci])
        if f_lag is None or f_lead is None:
            continue
        # The forecast target is the leader's actual current best score -- a
        # known quantity, held fixed across draws. Resampling it (the
        # within-resample max is stochastically below the observed max)
        # would bias the forward CI toward earlier catch-up.
        m = _trend_metrics(f_lag[0], f_lag[1], f_lead[0], f_lead[1], t_eval, leader_best_t)
        if m is None:
            continue
        boot_attempts += 1
        boot_backward.append(m["backward_months"])
        boot_velocity.append(m["velocity_months_per_year"])
        if m["forward_months"] is not None:
            boot_forward.append(m["forward_months"])
            boot_crossing.append(m["crossing_ord"])

    def _ci(vals: list[float]) -> Optional[list[float]]:
        if len(vals) < BOOT_MIN_VALID:
            return None
        lo, hi = np.quantile(np.array(vals), CI_QUANTILES)
        return [round(float(lo), 2), round(float(hi), 2)]

    def _ord_to_iso(o: float) -> Optional[str]:
        if not (1 <= o <= _MAX_ORDINAL):
            return None
        return datetime.fromordinal(int(round(o))).date().isoformat()

    # Forward CIs are only honest when nearly all otherwise-valid draws
    # produced a crossing; otherwise the interval silently conditions on
    # "catch-up happens at all".
    forward_ci_ok = (
        len(boot_forward) >= BOOT_MIN_VALID
        and boot_attempts > 0
        and len(boot_forward) >= FORWARD_CI_MIN_SHARE * boot_attempts
    )

    crossing_ci = None
    if forward_ci_ok:
        lo, hi = np.quantile(np.array(boot_crossing), CI_QUANTILES)
        lo_iso, hi_iso = _ord_to_iso(float(lo)), _ord_to_iso(float(hi))
        if lo_iso is not None and hi_iso is not None:
            crossing_ci = [lo_iso, hi_iso]

    return {
        "transform": transform,
        "as_of": pd.Timestamp(as_of_ts).isoformat(),
        "n_laggard": int(len(lag_ord)),
        "n_leader": int(len(lead_ord)),
        # Slopes in transform-space units per year (comparable within a
        # benchmark, not across transforms).
        "laggard_slope_per_year": round(m_lag * 365.25, 4),
        "leader_slope_per_year": round(m_lead * 365.25, 4),
        "backward_gap_months": round(point["backward_months"], 2),
        "backward_gap_ci_90": _ci(boot_backward),
        # Negative = the gap is shrinking.
        "gap_change_months_per_year": round(point["velocity_months_per_year"], 2),
        "gap_change_ci_90": _ci(boot_velocity),
        "forward_gap_months": (
            round(point["forward_months"], 2)
            if point["forward_months"] is not None
            else None
        ),
        "forward_gap_ci_90": _ci(boot_forward) if forward_ci_ok else None,
        "projected_catchup_date": (
            _ord_to_iso(point["crossing_ord"])
            if point["crossing_ord"] is not None
            else None
        ),
        "projected_catchup_ci_90": crossing_ci,
        "leader_current_best": {
            "model": leader.loc[lead_best_idx].get(
                model_col, leader.loc[lead_best_idx].get("model")
            ),
            "score": round(leader_best_score, 2),
        },
        "n_boot_valid": int(len(boot_backward)),
    }
