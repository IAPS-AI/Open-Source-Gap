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
    compute_trend_gap,
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
