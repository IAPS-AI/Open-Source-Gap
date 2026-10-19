"""Tests for scripts/gap_estimators.py (release-lag bracket + trend gap).

Run with: pytest tests/test_gap_estimators.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gap_estimators import (
    DAYS_PER_MONTH,
    compute_country_trend_lag,
    compute_trend_gap,
    country_frontier,
    current_lag_bracket,
    transform_scores,
)


def make_df(rows):
    df = pd.DataFrame(rows, columns=["model", "date", "score", "Open"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def months_between(a, b):
    return (pd.Timestamp(a) - pd.Timestamp(b)).days / DAYS_PER_MONTH


# ---------------------------------------------------------------------------
# transform_scores
# ---------------------------------------------------------------------------

def test_transform_linear_identity():
    v = np.array([1.0, 50.0, 199.0])
    assert np.allclose(transform_scores(v, "linear"), v)


def test_transform_logit_finite_at_bounds():
    v = np.array([0.0, 100.0, 50.0])
    t = transform_scores(v, "logit")
    assert np.all(np.isfinite(t))
    assert t[2] == pytest.approx(0.0)
    assert t[0] < 0 < t[1]


def test_transform_logit_monotone():
    v = np.array([10.0, 40.0, 60.0, 90.0])
    t = transform_scores(v, "logit")
    assert np.all(np.diff(t) > 0)


def test_transform_log():
    v = np.array([1.0, np.e])
    t = transform_scores(v, "log")
    assert t == pytest.approx([0.0, 1.0])


def test_transform_unknown_raises():
    with pytest.raises(ValueError):
        transform_scores(np.array([1.0]), "sqrt")


# ---------------------------------------------------------------------------
# current_lag_bracket
# ---------------------------------------------------------------------------

def test_bracket_basic_under_over_central():
    df = make_df([
        ("closed-a", "2025-01-01", 100.0, False),
        ("closed-b", "2025-07-01", 110.0, False),
        ("open-a", "2025-12-01", 105.0, True),
    ])
    as_of = "2026-01-01"
    b = current_lag_bracket(df, as_of=as_of)

    assert b["below_model"]["model"] == "closed-a"
    assert b["above_model"]["model"] == "closed-b"
    assert b["under_months"] == pytest.approx(months_between(as_of, "2025-07-01"), abs=0.01)
    assert b["over_months"] == pytest.approx(months_between(as_of, "2025-01-01"), abs=0.01)
    # Laggard level 105 sits exactly halfway between 100 and 110 (linear).
    span_days = (pd.Timestamp("2025-07-01") - pd.Timestamp("2025-01-01")).days
    cross = pd.Timestamp("2025-01-01") + pd.Timedelta(days=span_days * 0.5)
    expected_central = (pd.Timestamp(as_of) - cross).days / DAYS_PER_MONTH
    assert b["central_months"] == pytest.approx(expected_central, abs=0.05)
    # Bracket ordering: under <= central <= over.
    assert b["under_months"] <= b["central_months"] <= b["over_months"]
    assert not b["laggard_leads"]
    assert not b["censored"]


def test_bracket_laggard_leads():
    df = make_df([
        ("closed-a", "2025-01-01", 100.0, False),
        ("open-a", "2025-06-01", 120.0, True),
    ])
    b = current_lag_bracket(df, as_of="2026-01-01")
    assert b["laggard_leads"]
    assert b["under_months"] == 0.0
    assert b["over_months"] == 0.0
    assert b["central_months"] == 0.0


def test_bracket_censored_when_leader_started_above():
    df = make_df([
        ("closed-a", "2025-01-01", 100.0, False),
        ("open-a", "2025-06-01", 90.0, True),
    ])
    b = current_lag_bracket(df, as_of="2026-01-01")
    assert b["censored"]
    assert b["under_months"] == pytest.approx(
        months_between("2026-01-01", "2025-01-01"), abs=0.01)
    assert b["over_months"] is None
    assert b["central_months"] is None


def test_bracket_uses_leader_running_max_only():
    # closed-regression scores BELOW the earlier record; it must not become
    # the "below" anchor (running-max sequence skips it).
    df = make_df([
        ("closed-a", "2025-01-01", 100.0, False),
        ("closed-regression", "2025-03-01", 95.0, False),
        ("closed-b", "2025-07-01", 110.0, False),
        ("open-a", "2025-12-01", 105.0, True),
    ])
    b = current_lag_bracket(df, as_of="2026-01-01")
    assert b["below_model"]["model"] == "closed-a"
    assert b["above_model"]["model"] == "closed-b"


def test_bracket_logit_interpolation_differs_from_linear():
    df = make_df([
        ("closed-a", "2025-01-01", 80.0, False),
        ("closed-b", "2025-07-01", 95.0, False),
        ("open-a", "2025-12-01", 90.0, True),
    ])
    lin = current_lag_bracket(df, transform="linear", as_of="2026-01-01")
    lgt = current_lag_bracket(df, transform="logit", as_of="2026-01-01")
    # 90 is 2/3 of the way from 80 to 95 linearly, but less far along in
    # logit space (the 95 end is more compressed), so the logit crossing is
    # earlier -> larger central lag.
    assert lgt["central_months"] > lin["central_months"]
    # Bracket endpoints are transform-independent.
    assert lgt["under_months"] == lin["under_months"]
    assert lgt["over_months"] == lin["over_months"]


def test_bracket_none_when_group_missing():
    df = make_df([("closed-a", "2025-01-01", 100.0, False)])
    assert current_lag_bracket(df, as_of="2026-01-01") is None


# ---------------------------------------------------------------------------
# compute_trend_gap
# ---------------------------------------------------------------------------

def linear_series(start, n, score0, per_month, is_open, prefix):
    rows = []
    d = pd.Timestamp(start)
    for i in range(n):
        rows.append((f"{prefix}-{i}", d + pd.DateOffset(months=i),
                     score0 + per_month * i, is_open))
    return rows


def test_trend_gap_parallel_lines_recovers_offset():
    # Laggard follows the identical trajectory 6 months later.
    lead = linear_series("2024-01-01", 13, 100.0, 1.0, False, "closed")
    lag = [(m, d + pd.Timedelta(days=183), s, True)
           for (m, d, s, _) in linear_series("2024-01-01", 13, 100.0, 1.0, True, "open")]
    df = make_df(lead + lag)
    tg = compute_trend_gap(df, n_boot=200)

    assert tg is not None
    assert tg["backward_gap_months"] == pytest.approx(183 / DAYS_PER_MONTH, abs=0.15)
    assert tg["gap_change_months_per_year"] == pytest.approx(0.0, abs=0.1)
    # Laggard line reaches the leader's best observed score right at as_of.
    assert tg["forward_gap_months"] == pytest.approx(0.0, abs=0.15)
    assert tg["n_laggard"] == 13
    assert tg["n_leader"] == 13


def test_trend_gap_converging_velocity_negative():
    lead = linear_series("2024-01-01", 13, 100.0, 1.0, False, "closed")
    lag = linear_series("2024-01-01", 13, 88.0, 2.0, True, "open")
    df = make_df(lead + lag)
    tg = compute_trend_gap(df, n_boot=200)

    # d(gap)/dt = 1 - 2/1 = -1 day/day -> -12 months/year.
    assert tg["gap_change_months_per_year"] == pytest.approx(-12.0, abs=0.5)
    assert tg["gap_change_months_per_year"] < 0
    assert tg["forward_gap_months"] is not None


def test_trend_gap_diverging_velocity_positive():
    lead = linear_series("2024-01-01", 13, 100.0, 2.0, False, "closed")
    lag = linear_series("2024-01-01", 13, 96.0, 1.0, True, "open")
    df = make_df(lead + lag)
    tg = compute_trend_gap(df, n_boot=200)
    # d(gap)/dt = 1 - 1/2 = +0.5 day/day -> +6 months/year.
    assert tg["gap_change_months_per_year"] == pytest.approx(6.0, abs=0.5)


def test_trend_gap_none_when_leader_flat():
    lead = linear_series("2024-01-01", 6, 100.0, 0.0, False, "closed")
    lag = linear_series("2024-01-01", 6, 90.0, 1.0, True, "open")
    assert compute_trend_gap(make_df(lead + lag), n_boot=50) is None


def test_trend_gap_none_when_too_few_points():
    lead = linear_series("2024-01-01", 2, 100.0, 1.0, False, "closed")
    lag = linear_series("2024-01-01", 6, 90.0, 1.0, True, "open")
    assert compute_trend_gap(make_df(lead + lag), n_boot=50) is None


def test_trend_gap_deterministic_for_seed():
    rng = np.random.default_rng(42)
    lead = [(m, d, s + rng.normal(0, 0.5), o)
            for (m, d, s, o) in linear_series("2024-01-01", 13, 100.0, 1.0, False, "closed")]
    lag = [(m, d, s + rng.normal(0, 0.5), o)
           for (m, d, s, o) in linear_series("2024-01-01", 13, 92.0, 1.0, True, "open")]
    df = make_df(lead + lag)
    a = compute_trend_gap(df, n_boot=300, seed=7)
    b = compute_trend_gap(df, n_boot=300, seed=7)
    assert a == b


def test_trend_gap_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    lead = [(m, d, s + rng.normal(0, 0.5), o)
            for (m, d, s, o) in linear_series("2024-01-01", 15, 100.0, 1.0, False, "closed")]
    lag = [(m, d, s + rng.normal(0, 0.5), o)
           for (m, d, s, o) in linear_series("2024-01-01", 15, 92.0, 1.0, True, "open")]
    tg = compute_trend_gap(make_df(lead + lag), n_boot=500)

    lo, hi = tg["backward_gap_ci_90"]
    assert lo <= tg["backward_gap_months"] <= hi
    lo_v, hi_v = tg["gap_change_ci_90"]
    assert lo_v <= tg["gap_change_months_per_year"] <= hi_v
    assert tg["n_boot_valid"] > 400


def test_trend_gap_log_transform_metr_style():
    # Horizons double every 4 months in both groups; laggard is 8 months
    # behind -> backward gap ~8 months regardless of the doubling scale.
    def horizon_series(start, n, h0, is_open, prefix):
        rows = []
        d = pd.Timestamp(start)
        for i in range(n):
            rows.append((f"{prefix}-{i}", d + pd.DateOffset(months=i),
                         h0 * (2 ** (i / 4.0)), is_open))
        return rows

    lead = horizon_series("2024-01-01", 13, 10.0, False, "closed")
    lag = horizon_series("2024-09-01", 13, 10.0, True, "open")
    tg = compute_trend_gap(make_df(lead + lag), transform="log", n_boot=200)

    assert tg is not None
    assert tg["backward_gap_months"] == pytest.approx(8.0, abs=0.4)


def test_bracket_default_as_of_is_data_max_not_wall_clock():
    df = make_df([
        ("closed-a", "2025-01-01", 100.0, False),
        ("closed-b", "2025-07-01", 110.0, False),
        ("open-a", "2025-12-01", 105.0, True),
    ])
    b = current_lag_bracket(df)
    # Reproducible: evaluated at the latest release date, not at runtime now().
    assert b["as_of"].startswith("2025-12-01")
    assert b["under_months"] == pytest.approx(
        months_between("2025-12-01", "2025-07-01"), abs=0.01)


def test_trend_gap_near_flat_laggard_does_not_crash():
    # A barely-positive laggard slope projects the crossing centuries out;
    # must return None projections, not raise from datetime.fromordinal.
    lead = linear_series("2024-01-01", 13, 100.0, 2.0, False, "closed")
    lag = [(f"open-{i}", pd.Timestamp("2024-01-01") + pd.DateOffset(months=i),
            30.0 + 0.0001 * i, True) for i in range(13)]
    tg = compute_trend_gap(make_df(lead + lag), n_boot=200)

    assert tg is not None
    assert tg["forward_gap_months"] is None
    assert tg["projected_catchup_date"] is None
    assert tg["projected_catchup_ci_90"] is None
    import json
    json.dumps(tg)


def test_trend_gap_ci_suppressed_for_small_groups():
    # 5 models per group: point estimates published, CIs suppressed (a
    # percentile bootstrap over so few records has no 90% coverage).
    lead = linear_series("2024-01-01", 5, 100.0, 1.0, False, "closed")
    lag = linear_series("2024-01-01", 5, 92.0, 1.0, True, "open")
    tg = compute_trend_gap(make_df(lead + lag), n_boot=500)

    assert tg is not None
    assert tg["backward_gap_months"] is not None
    assert tg["backward_gap_ci_90"] is None
    assert tg["gap_change_ci_90"] is None
    assert tg["forward_gap_ci_90"] is None
    assert tg["projected_catchup_ci_90"] is None
    assert tg["n_boot_valid"] == 0


def test_trend_gap_reported_fields_json_safe():
    lead = linear_series("2024-01-01", 13, 100.0, 1.0, False, "closed")
    lag = linear_series("2024-01-01", 13, 92.0, 1.0, True, "open")
    tg = compute_trend_gap(make_df(lead + lag), n_boot=200)
    import json
    json.dumps(tg)  # must not raise (no numpy scalars / inf / NaN)
    assert tg["projected_catchup_date"] is not None


# ---------------------------------------------------------------------------
# compute_country_trend_lag (US-China appendix method)
# ---------------------------------------------------------------------------

def country_series(prefix, start, n, step_days, score0, per_day, is_china):
    base = pd.Timestamp(start)
    return [
        (f"{prefix}{i}", base + pd.Timedelta(days=step_days * i),
         score0 + per_day * step_days * i, is_china)
        for i in range(n)
    ]


def test_country_lag_parallel_lines_recovers_offset():
    # China runs the US trajectory exactly 183 days later.
    us = country_series("US", "2024-01-01", 10, 60, 100.0, 0.05, False)
    cn = country_series("CN", "2024-07-02", 8, 60, 100.0, 0.05, True)
    r = compute_country_trend_lag(make_df(us + cn), transform="linear")
    assert r["lag_months"] == pytest.approx(183 / DAYS_PER_MONTH, abs=0.01)
    assert r["gap_change_months_per_year"] == pytest.approx(0.0, abs=1e-6)
    assert r["laggard"]["n_points"] == 8 and r["leader"]["n_points"] == 10
    assert r["laggard"]["r2"] == pytest.approx(1.0)
    assert r["leader"]["slope_per_month"] == pytest.approx(0.05 * DAYS_PER_MONTH, abs=1e-3)


def test_country_lag_evaluated_at_latest_laggard_frontier_point():
    # A later Chinese release below the record is not a frontier point, so
    # it does not move the evaluation date.
    us = country_series("US", "2024-01-01", 10, 60, 100.0, 0.05, False)
    cn = country_series("CN", "2024-07-02", 8, 60, 100.0, 0.05, True)
    cn.append(("CN-small", pd.Timestamp("2026-01-01"), 90.0, True))
    r = compute_country_trend_lag(make_df(us + cn), transform="linear")
    assert r["eval_model"] == "CN7"
    assert r["eval_date"] == (pd.Timestamp("2024-07-02") + pd.Timedelta(days=420)).date().isoformat()
    assert r["laggard"]["n_points"] == 8


def test_country_lag_uses_trend_level_not_model_score():
    # The last Chinese point sits above its own trend: the lag reads the
    # trend level, not that model's score.
    us = country_series("US", "2024-01-01", 10, 60, 100.0, 0.05, False)
    cn = country_series("CN", "2024-07-02", 7, 60, 100.0, 0.05, True)
    cn.append(("CNjump", pd.Timestamp("2024-07-02") + pd.Timedelta(days=420),
               100.0 + 0.05 * 420 + 3.0, True))
    r = compute_country_trend_lag(make_df(us + cn), transform="linear")
    assert r["laggard_trend_level"] < r["eval_model_score"]
    # Lag if measured at the model's own score: 3 points less of US progress.
    at_score_lag = (183 - 3.0 / 0.05) / DAYS_PER_MONTH
    assert r["lag_months"] > at_score_lag + 0.5


def test_country_frontier_collapses_same_day_and_is_strict():
    df = make_df([
        ("a", "2024-01-01", 10.0, False),
        ("b", "2024-02-01", 12.0, False),
        ("c", "2024-02-01", 15.0, False),   # same day, higher: the day's max
        ("d", "2024-03-01", 15.0, False),   # ties the record: not a record
        ("e", "2024-04-01", 11.0, False),   # below the record
        ("f", "2024-05-01", 16.0, False),
    ])
    f = country_frontier(df, "score", "model")
    assert f["model"].tolist() == ["a", "c", "f"]


def test_country_lag_window_cut_before_frontier():
    # A strong pre-2024 US model must not suppress the 2024 US frontier.
    us = [("US-old", pd.Timestamp("2023-06-01"), 500.0, False)]
    us += country_series("US", "2024-01-01", 10, 60, 100.0, 0.05, False)
    cn = country_series("CN", "2024-07-02", 8, 60, 100.0, 0.05, True)
    r = compute_country_trend_lag(make_df(us + cn), transform="linear")
    assert r["window_start"] == "2024-01-01"
    assert r["leader"]["n_points"] == 10
    assert r["leader"]["first_model"] == "US0"
    assert r["lag_months"] == pytest.approx(183 / DAYS_PER_MONTH, abs=0.01)


def test_country_lag_log_transform_recovers_offset():
    # Horizons doubling every 213 days; China 182 days behind.
    def series(prefix, start, n, is_china):
        base = pd.Timestamp(start)
        return [(f"{prefix}{i}", base + pd.Timedelta(days=50 * i),
                 2.0 ** ((50 * i) / 213.0), is_china) for i in range(n)]
    us = series("US", "2024-01-01", 12, False)
    cn = series("CN", "2024-07-01", 8, True)
    r = compute_country_trend_lag(make_df(us + cn), transform="log")
    assert r["lag_months"] == pytest.approx(182 / DAYS_PER_MONTH, abs=0.01)


def test_country_lag_none_when_too_few_points_or_flat_leader():
    us = country_series("US", "2024-01-01", 10, 60, 100.0, 0.05, False)
    cn = country_series("CN", "2024-07-02", 2, 60, 100.0, 0.05, True)
    assert compute_country_trend_lag(make_df(us + cn)) is None
    # A flat US series has a one-point strict frontier -> None.
    flat = [(f"US{i}", pd.Timestamp("2024-01-01") + pd.Timedelta(days=30 * i), 100.0, False)
            for i in range(5)]
    cn4 = country_series("CN", "2024-07-02", 4, 60, 90.0, 0.05, True)
    assert compute_country_trend_lag(make_df(flat + cn4)) is None


def test_country_lag_publishes_points_and_trendline_for_plotting():
    us = country_series("US", "2024-01-01", 10, 60, 100.0, 0.05, False)
    cn = country_series("CN", "2024-07-02", 8, 60, 100.0, 0.05, True)
    r = compute_country_trend_lag(make_df(us + cn), transform="linear")
    lead, lag = r["leader"], r["laggard"]
    assert [p["model"] for p in lag["points"]] == [f"CN{i}" for i in range(8)]
    assert len(lead["points"]) == lead["n_points"]
    # Lines start at each frontier's first point and share an end date.
    assert lead["trendline"][0]["date"] == lead["first_date"]
    assert lag["trendline"][0]["date"] == lag["first_date"]
    assert lead["trendline"][-1]["date"] == lag["trendline"][-1]["date"] == lag["last_date"]
    # Exact linear data: the line passes through the points.
    assert lag["trendline"][0]["score"] == pytest.approx(lag["points"][0]["score"], abs=1e-6)
    # The lag marker sits on both lines: the US line at the reach date and
    # the China line at the evaluation date share the laggard trend level.
    assert r["laggard_trend_level"] == pytest.approx(lag["points"][-1]["score"], abs=0.01)


def test_country_lag_logit_trendline_in_raw_percent():
    def series(prefix, start, n, is_china):
        base = pd.Timestamp(start)
        return [(f"{prefix}{i}", base + pd.Timedelta(days=60 * i),
                 100.0 / (1.0 + np.exp(-(-2.0 + 0.004 * 60 * i))), is_china)
                for i in range(n)]
    r = compute_country_trend_lag(
        make_df(series("US", "2024-01-01", 10, False) + series("CN", "2024-07-01", 8, True)),
        transform="logit")
    scores = [p["score"] for p in r["leader"]["trendline"]]
    assert all(0 < s < 100 for s in scores)
    assert scores == sorted(scores)


def test_country_lag_json_safe():
    us = country_series("US", "2024-01-01", 10, 60, 100.0, 0.05, False)
    cn = country_series("CN", "2024-07-02", 8, 60, 100.0, 0.05, True)
    import json
    json.dumps(compute_country_trend_lag(make_df(us + cn)))


# ECI frontier points from 2024 onward as published in data.json on
# 2026-09-13: the data vintage behind the appendix's "20 US / 17 China
# frontier-defining points ... 1.31 vs 1.18 ECI/month ... roughly 6.5 months".
APPENDIX_ECI_FRONTIER = [
    ("GPT-4 Turbo (Nov 2023)", "2024-01-25", 126.45, False),
    ("Claude 3 Opus", "2024-02-29", 126.91, False),
    ("GPT-4 Turbo (Apr 2024)", "2024-04-09", 127.25, False),
    ("GPT-4o", "2024-05-13", 128.97, False),
    ("Claude 3.5 Sonnet", "2024-06-20", 130.0, False),
    ("o1-mini", "2024-09-12", 135.83, False),
    ("o1", "2024-12-17", 141.87, False),
    ("Gemini 2.5 Pro (Mar 2025)", "2025-03-31", 144.17, False),
    ("o3", "2025-04-16", 146.91, False),
    ("o3-pro", "2025-06-10", 147.45, False),
    ("GPT-5", "2025-08-07", 150.0, False),
    ("GPT-5 Pro", "2025-10-07", 150.28, False),
    ("Gemini 3 Pro", "2025-11-18", 153.01, False),
    ("GPT-5.2 Pro", "2025-12-11", 155.34, False),
    ("GPT-5.3 Codex", "2026-02-05", 156.59, False),
    ("GPT-5.4 Pro", "2026-03-05", 158.93, False),
    ("GPT-5.5 Pro", "2026-04-23", 162.26, False),
    ("Claude Fable 5", "2026-06-09", 163.41, False),
    ("Claude Fable 5.1", "2026-09-01", 164.24, False),
    ("GPT-6 Astra", "2026-09-03", 166.57, False),
    ("DeepSeek-V2 (MoE-236B)", "2024-05-07", 124.77, True),
    ("Qwen2-72B", "2024-06-07", 125.27, True),
    ("Qwen2.5-32B", "2024-09-17", 128.52, True),
    ("Qwen2.5-72B", "2024-09-19", 129.0, True),
    ("DeepSeek-V3", "2024-12-26", 132.35, True),
    ("DeepSeek-R1", "2025-01-20", 138.98, True),
    ("Qwen3-235B-A22B", "2025-04-29", 139.36, True),
    ("DeepSeek-R1 (May 2025)", "2025-05-28", 141.3, True),
    ("Qwen3-235B-A22B (Jul 2025)", "2025-07-25", 143.88, True),
    ("DeepSeek-V3.2-Exp", "2025-09-29", 145.03, True),
    ("Kimi K2 Thinking", "2025-11-06", 145.77, True),
    ("DeepSeek-V3.2", "2025-12-01", 146.2, True),
    ("Kimi K2.5", "2026-01-27", 148.03, True),
    ("GLM-5.1", "2026-04-07", 149.73, True),
    ("Kimi K2.6", "2026-04-20", 151.01, True),
    ("Qwen3.7-Max", "2026-05-19", 153.72, True),
    ("Kimi K3", "2026-07-16", 157.64, True),
]


def test_country_lag_reproduces_appendix_eci_figures():
    r = compute_country_trend_lag(make_df(APPENDIX_ECI_FRONTIER), transform="linear")
    assert r["leader"]["n_points"] == 20
    assert r["laggard"]["n_points"] == 17
    assert r["leader"]["slope_per_month"] == pytest.approx(1.31, abs=0.005)
    assert r["laggard"]["slope_per_month"] == pytest.approx(1.18, abs=0.005)
    assert r["leader"]["r2"] == pytest.approx(0.99, abs=0.005)
    assert r["laggard"]["r2"] == pytest.approx(0.97, abs=0.005)
    assert r["eval_model"] == "Kimi K3"
    assert r["lag_months"] == pytest.approx(6.5, abs=0.05)
    assert r["gap_change_months_per_year"] > 0  # US pulling ahead slightly
