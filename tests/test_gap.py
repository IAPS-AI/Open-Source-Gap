"""The gap calculation: frontiers, the caught-up test, the daily series, the
headline window, the rolling average, the per-release matches, and the two
published comparisons."""
import json
import random
from datetime import date, timedelta

import numpy as np
import pytest

from gap import (
    CHINA,
    DAYS_PER_MONTH,
    UNITED_STATES,
    Criterion,
    Model,
    build_comparison,
    build_payload,
    build_segments,
    daily,
    firm_from,
    first_matches,
    headline_window,
    months_before,
    quantile,
    record_setters,
    rolling_average,
    segment_on,
    summarize,
    trend_lag,
)


def closed(name, day, eci, ci=None, country=UNITED_STATES):
    return _model(name, day, eci, False, ci, country)


def opened(name, day, eci, ci=None, country=CHINA):
    return _model(name, day, eci, True, ci, country)


def _model(name, day, eci, is_open, ci, country):
    low, high = (eci - ci, eci + ci) if ci is not None else (None, None)
    return Model(name, date.fromisoformat(day), eci, is_open, low, high, country=country)


class FakeDraws:
    """Stands in for EciBootstrap: fixed shares for named pairs."""

    n_samples = 500

    def __init__(self, shares):
        self.shares = shares

    def prob_exceeds(self, a, b):
        return self.shares.get((a, b))


def months(later, earlier):
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days / DAYS_PER_MONTH


def series(models, as_of, draws=None):
    laggards = record_setters(m for m in models if m.is_open)
    leaders = record_setters(m for m in models if not m.is_open)
    return build_segments(laggards, leaders, Criterion(draws), date.fromisoformat(as_of))


# ---------------------------------------------------------------- frontier

def test_record_setters_keep_only_new_records_in_release_order():
    models = [closed("C", "2024-03-01", 110), closed("A", "2024-01-01", 100),
              closed("B", "2024-02-01", 95), closed("D", "2024-04-01", 110)]
    assert [m.name for m in record_setters(models)] == ["A", "C"]  # B is lower, D only ties


def test_only_the_best_model_released_on_a_day_can_set_the_record():
    models = [closed("low", "2024-01-01", 100), closed("high", "2024-01-01", 105)]
    assert [m.name for m in record_setters(models)] == ["high"]


# --------------------------------------------------------------- criterion

def test_bootstrap_share_decides_when_both_models_are_in_the_fit():
    a, b = opened("a", "2024-02-01", 100, ci=1), closed("b", "2024-01-01", 120, ci=1)
    # 20 points behind, but ahead in 5% of paired resamples: caught up.
    assert Criterion(FakeDraws({("a", "b"): 0.05})).test(a, b) == (True, "bootstrap")
    assert Criterion(FakeDraws({("a", "b"): 0.049})).test(a, b) == (False, "bootstrap")


def test_analytical_test_is_used_when_a_model_is_missing_from_the_fit():
    record = closed("b", "2024-01-01", 120, ci=1.645)          # sd 1
    near = opened("near", "2024-02-01", 117.7, ci=1.645)       # 2.3 behind, limit 1.645*sqrt(2) = 2.33
    far = opened("far", "2024-02-01", 117.6, ci=1.645)         # 2.4 behind
    criterion = Criterion(FakeDraws({}))
    assert criterion.test(near, record) == (True, "analytical")
    assert criterion.test(far, record) == (False, "analytical")


def test_threshold_is_the_last_resort_without_intervals():
    record = closed("b", "2024-01-01", 120)
    assert Criterion().test(opened("a", "2024-02-01", 119.0), record) == (True, "threshold")
    assert Criterion().test(opened("c", "2024-02-01", 118.9), record) == (False, "threshold")


def test_a_zero_width_interval_on_one_side_still_allows_the_analytical_test():
    anchor = closed("anchor", "2024-01-01", 130.0, ci=0.0)     # anchors are published without spread
    rival = opened("rival", "2024-02-01", 128.5, ci=1.645)
    assert Criterion().test(rival, anchor) == (True, "analytical")


# ------------------------------------------------------------ daily series

def test_gap_counts_from_the_first_leader_record_until_one_is_matched():
    models = [closed("C1", "2024-01-01", 120), opened("O1", "2024-03-01", 100)]
    (only,) = series(models, "2024-06-01")
    assert only.start == date(2024, 3, 1)                     # first day both groups exist
    assert (only.reference.name, only.matched, only.unmatched.name) == ("C1", False, "C1")
    assert only.months(date(2024, 6, 1)) == pytest.approx(months("2024-06-01", "2024-01-01"))


def test_gap_drops_when_a_laggard_catches_a_newer_leader_record():
    models = [
        closed("C1", "2024-01-01", 120), closed("C2", "2024-05-01", 130),
        opened("O1", "2024-02-01", 119.5),                    # within a point of C1
        opened("O2", "2024-08-01", 129.5),                    # within a point of C2
    ]
    segs = series(models, "2024-10-01")
    assert [(s.start.isoformat(), s.laggard.name, s.reference.name) for s in segs] == [
        ("2024-02-01", "O1", "C1"), ("2024-05-01", "O1", "C1"), ("2024-08-01", "O2", "C2")]
    # The leader release on May 1 changes the frontier but not the reference.
    assert segs[1].leader.name == "C2" and segs[1].unmatched.name == "C2"
    before = segment_on(segs, date(2024, 7, 31)).months(date(2024, 7, 31))
    after = segment_on(segs, date(2024, 8, 1)).months(date(2024, 8, 1))
    assert before == pytest.approx(months("2024-07-31", "2024-01-01"))
    assert after == pytest.approx(months("2024-08-01", "2024-05-01"))
    assert segs[-1].unmatched is None                          # caught up to the newest record


def test_gap_is_zero_on_the_day_a_leader_record_arrives_already_matched():
    models = [closed("C1", "2024-01-01", 120), opened("O1", "2024-02-01", 125),
              closed("C2", "2024-06-01", 125.5)]
    segs = series(models, "2024-07-01")
    on_release = segment_on(segs, date(2024, 6, 1))
    assert on_release.reference.name == "C2"
    assert on_release.months(date(2024, 6, 1)) == 0.0


def test_a_new_laggard_record_can_fail_a_match_its_predecessor_passed():
    models = [closed("C1", "2024-01-01", 120), closed("C2", "2024-03-01", 130),
              opened("O1", "2024-04-01", 128), opened("O2", "2024-06-01", 128.5)]
    draws = FakeDraws({("O1", "C1"): 1.0, ("O1", "C2"): 0.10,
                       ("O2", "C1"): 1.0, ("O2", "C2"): 0.01})
    segs = series(models, "2024-07-01", draws)
    assert [(s.laggard.name, s.reference.name) for s in segs] == [("O1", "C2"), ("O2", "C1")]
    assert segs[1].unmatched.name == "C2"


def test_series_needs_both_groups_and_a_day_they_overlap():
    with pytest.raises(ValueError):
        series([closed("C1", "2024-01-01", 120)], "2024-06-01")
    with pytest.raises(ValueError):
        series([closed("C1", "2024-01-01", 120), opened("O1", "2024-09-01", 100)], "2024-06-01")


def naive_gap(models, day, criterion):
    """Epoch's description, followed literally for one day."""
    available = [m for m in models if m.released <= day]
    best_open = max((m for m in available if m.is_open),
                    key=lambda m: (m.eci, -m.released.toordinal()))
    records, top = [], float("-inf")
    for m in sorted((m for m in available if not m.is_open), key=lambda m: (m.released, -m.eci)):
        if m.eci > top:
            top = m.eci
            records.append(m)
    reference = records[0]
    for record in reversed(records):
        if criterion.caught_up(best_open, record):
            reference = record
            break
    time_gap = (day - reference.released).days / DAYS_PER_MONTH
    return time_gap, max(records[-1].eci - best_open.eci, 0.0)


def random_models(seed):
    rng = random.Random(seed)
    start = date(2023, 1, 1)
    models = []
    for i in range(rng.randint(8, 30)):
        is_open = rng.random() < 0.5
        models.append(Model(f"{'o' if is_open else 'c'}{i}", start + timedelta(days=rng.randint(0, 900)),
                            round(rng.uniform(100, 160), 2), is_open))
    models.append(Model("c-first", start, 100.0, False))
    models.append(Model("o-first", start + timedelta(days=5), 90.0, True))
    # Random but fixed paired shares, so matches are not monotone in the score.
    shares = {(a.name, b.name): rng.random() * 0.2 for a in models for b in models}
    return models, shares, start + timedelta(days=1000)


@pytest.mark.parametrize("seed", range(12))
def test_segments_reproduce_a_literal_day_by_day_calculation(seed):
    models, shares, as_of = random_models(seed)
    for draws in (None, FakeDraws(shares)):
        criterion = Criterion(draws)
        laggards = record_setters(m for m in models if m.is_open)
        leaders = record_setters(m for m in models if not m.is_open)
        segs = build_segments(laggards, leaders, criterion, as_of)
        checked = 0
        for day, seg in daily(segs, segs[0].start, as_of):
            want_months, want_points = naive_gap(models, day, criterion)
            assert seg.months(day) == pytest.approx(want_months), day
            assert max(seg.eci_points, 0.0) == pytest.approx(want_points), day
            checked += 1
        assert checked == (as_of - segs[0].start).days + 1


# ------------------------------------------------------------- statistics

def test_quantile_matches_numpy():
    rng = random.Random(1)
    for n in (1, 2, 7, 366):
        values = [rng.uniform(0, 15) for _ in range(n)]
        for q in (0.0, 0.05, 0.5, 0.95, 1.0):
            assert quantile(values, q) == pytest.approx(float(np.quantile(values, q)))


def test_months_before_handles_month_ends():
    assert months_before(date(2026, 11, 9), 12) == date(2025, 11, 9)
    assert months_before(date(2024, 2, 29), 12) == date(2023, 2, 28)
    assert months_before(date(2025, 3, 31), 1) == date(2025, 2, 28)
    assert months_before(date(2025, 1, 15), 2) == date(2024, 11, 15)


def test_summarize_averages_the_daily_values():
    models = [closed("C1", "2024-01-01", 120), opened("O1", "2024-01-11", 110)]
    segs = series(models, "2024-01-20")
    out = summarize(segs, date(2024, 1, 11), date(2024, 1, 20))
    daily_months = [d / DAYS_PER_MONTH for d in range(10, 20)]   # 10..19 days since C1
    assert out["days"] == 10
    assert out["months"] == pytest.approx(sum(daily_months) / 10)
    assert out["months_p05"] == pytest.approx(float(np.quantile(daily_months, 0.05)))
    assert out["months_p95"] == pytest.approx(float(np.quantile(daily_months, 0.95)))
    assert out["eci_points"] == pytest.approx(10.0)


def test_eci_gap_is_floored_at_zero_when_a_laggard_leads():
    models = [closed("C1", "2024-01-01", 120), opened("O1", "2024-01-11", 125)]
    segs = series(models, "2024-01-20")
    assert segs[0].eci_points == pytest.approx(-5.0)             # the segment keeps the sign
    assert summarize(segs, date(2024, 1, 11), date(2024, 1, 20))["eci_points"] == 0.0


def test_headline_window_is_the_year_to_the_latest_record_setting_release():
    models = [closed("C1", "2023-01-01", 120), opened("O1", "2023-02-01", 110),
              closed("C2", "2025-03-10", 140), opened("late-but-lower", "2025-06-01", 100)]
    laggards = record_setters(m for m in models if m.is_open)
    leaders = record_setters(m for m in models if not m.is_open)
    as_of = date(2025, 9, 1)
    segs = build_segments(laggards, leaders, Criterion(), as_of)
    # Ends at C2 (the last record), not at as_of and not at the non-record open model.
    assert headline_window(segs, laggards, leaders, as_of) == (date(2024, 3, 10), date(2025, 3, 10), False)


def test_headline_window_uses_all_history_when_there_is_under_a_year():
    models = [closed("C1", "2025-01-01", 120), opened("O1", "2025-02-01", 110),
              closed("C2", "2025-08-01", 140)]
    laggards = record_setters(m for m in models if m.is_open)
    leaders = record_setters(m for m in models if not m.is_open)
    as_of = date(2025, 9, 1)
    segs = build_segments(laggards, leaders, Criterion(), as_of)
    assert headline_window(segs, laggards, leaders, as_of) == (date(2025, 2, 1), date(2025, 8, 1), True)


# -------------------------------------------------------- rolling average

@pytest.mark.parametrize("seed", range(6))
def test_rolling_average_is_the_mean_of_the_prior_twelve_months_each_day(seed):
    models, shares, as_of = random_models(seed)
    segs = build_segments(record_setters(m for m in models if m.is_open),
                          record_setters(m for m in models if not m.is_open),
                          Criterion(FakeDraws(shares)), as_of)
    first, values = rolling_average(segs, as_of)
    by_day = {day: seg.months(day) for day, seg in daily(segs, segs[0].start, as_of)}

    assert months_before(first, 12) > segs[0].start                 # a full window behind it
    assert months_before(first - timedelta(days=1), 12) <= segs[0].start
    assert len(values) == (as_of - first).days + 1
    for offset in range(0, len(values), 17):                         # spot-check through the series
        day = first + timedelta(days=offset)
        window = [by_day[months_before(day, 12) + timedelta(days=i)]
                  for i in range((day - months_before(day, 12)).days + 1)]
        assert values[offset] == pytest.approx(sum(window) / len(window)), day


def test_rolling_average_is_empty_without_a_full_year():
    segs = series([closed("C1", "2025-01-01", 120), opened("O1", "2025-02-01", 110)], "2025-12-31")
    assert rolling_average(segs, date(2025, 12, 31)) == (None, [])


def test_rolling_average_becomes_firm_a_year_after_the_first_match():
    models = [closed("C1", "2023-01-01", 120), opened("O1", "2023-02-01", 100),
              opened("O2", "2024-06-15", 125)]                       # first match: June 15, 2024
    segs = series(models, "2026-01-01")
    first, _ = rolling_average(segs, date(2026, 1, 1))
    assert first == date(2024, 2, 2)                                 # one year and a day after Feb 1, 2023
    assert firm_from(segs, first, date(2026, 1, 1)) == date(2025, 6, 15)
    assert firm_from(segs, first, date(2025, 6, 14)) is None         # not reached yet
    never = series(models[:2], "2026-01-01")
    assert firm_from(never, date(2024, 2, 2), date(2026, 1, 1)) is None


# ---------------------------------------------------------------- matches

def test_first_match_is_the_first_laggard_record_to_catch_up():
    laggards = [opened("O1", "2024-02-01", 110), opened("O2", "2024-06-01", 119.2),
                opened("O3", "2024-09-01", 125)]
    leaders = [closed("C1", "2024-01-01", 120), closed("C2", "2024-07-01", 140)]
    first, second = first_matches(laggards, leaders, Criterion())
    assert (first.by.name, first.on, first.test) == ("O2", date(2024, 6, 1), "threshold")
    assert first.lag_months == pytest.approx(months("2024-06-01", "2024-01-01"))
    assert (second.by, second.on, second.lag_months, second.test) == (None, None, None, None)


def test_a_record_the_best_laggard_already_matches_has_no_lag():
    laggards = [opened("O1", "2024-02-01", 125)]
    leaders = [closed("C1", "2024-01-01", 120), closed("C2", "2024-06-01", 125.5)]
    _, second = first_matches(laggards, leaders, Criterion())
    assert (second.by.name, second.on, second.lag_months) == ("O1", date(2024, 6, 1), 0.0)


# ---------------------------------------------------------------- payload

def payload_models():
    """Closed models are American and open ones Chinese, so both comparisons
    split these the same way."""
    return [
        closed("C1", "2023-01-01", 120, ci=2), closed("C2", "2024-05-01", 130, ci=2),
        closed("C3", "2025-01-15", 142.346, ci=2),
        opened("O1", "2023-02-01", 119.5, ci=2), opened("O2", "2024-08-01", 129.5, ci=2),
        opened("unreleased-yet", "2030-01-01", 200, ci=2),
    ]


def test_payload_is_plain_json_with_both_comparisons():
    payload = build_payload(payload_models(), date(2025, 6, 30))
    assert json.loads(json.dumps(payload)) == payload
    assert set(payload) == {"schema", "as_of", "days_per_month", "test", "comparisons"}
    assert payload["schema"] == 2 and payload["as_of"] == "2025-06-30"
    assert payload["test"] == {"paired_bootstrap": False, "resamples": None, "min_share": 0.05}
    assert list(payload["comparisons"]) == ["open_closed", "china_us"]
    for comparison in payload["comparisons"].values():
        assert set(comparison) == {"headline", "today", "segments", "average", "leader_frontier",
                                   "laggard_frontier", "models_scored", "trend"}
        assert comparison["trend"] is None                                   # too few records since 2024
        assert comparison["models_scored"] == {"laggard": 2, "leader": 3}   # the 2030 model is not out yet
    assert payload["comparisons"]["open_closed"] == payload["comparisons"]["china_us"]


def test_comparison_today_matches_the_last_segment_and_frontier_lists():
    comparison = build_payload(payload_models(), date(2025, 6, 30))["comparisons"]["open_closed"]
    assert comparison["today"] == {
        "months": round(months("2025-06-30", "2024-05-01"), 1),
        "eci_points": round(142.346 - 129.5, 1),
        "laggard_model": "O2", "leader_model": "C3", "reference_model": "C2",
        "matched": True, "unmatched_model": "C3",
    }
    assert comparison["segments"][-1] == {
        "start": "2025-01-15", "laggard_model": "O2", "leader_model": "C3",
        "reference_model": "C2", "matched": True, "unmatched_model": "C3"}
    assert [m["model"] for m in comparison["laggard_frontier"]] == ["O1", "O2"]
    c1, c2, c3 = comparison["leader_frontier"]
    assert c3 == {"model": "C3", "released": "2025-01-15", "eci": 142.35, "ci_low": 140.35,
                  "ci_high": 144.35, "organization": "", "matched_by": None,
                  "matched_on": None, "lag_months": None, "test": None}
    assert (c1["matched_by"], c1["matched_on"], c1["lag_months"]) == ("O1", "2023-02-01", 1.0)
    assert (c2["matched_by"], c2["lag_months"]) == ("O2", round(months("2024-08-01", "2024-05-01"), 1))


def test_comparison_headline_is_the_trailing_year_average():
    comparison = build_payload(payload_models(), date(2025, 6, 30))["comparisons"]["open_closed"]
    headline = comparison["headline"]
    assert (headline["window_start"], headline["window_end"]) == ("2024-01-15", "2025-01-15")
    assert headline["days"] == 367 and headline["full_history"] is False
    # Day by day, by hand: O1 against C1 until O2 arrives on Aug 1, then O2 against C2.
    values = []
    day = date(2024, 1, 15)
    while day <= date(2025, 1, 15):
        ref = date(2023, 1, 1) if day < date(2024, 8, 1) else date(2024, 5, 1)
        values.append((day - ref).days / DAYS_PER_MONTH)
        day += timedelta(days=1)
    assert headline["months"] == round(sum(values) / len(values), 1)
    assert headline["months_p95"] == round(float(np.quantile(values, 0.95)), 1)


def test_the_rolling_average_ends_on_the_headline():
    comparison = build_payload(payload_models(), date(2025, 6, 30))["comparisons"]["open_closed"]
    average, headline = comparison["average"], comparison["headline"]
    first = date.fromisoformat(average["start"])
    assert first == date(2024, 2, 2)                                 # a full year after the series starts
    assert first + timedelta(days=len(average["months"]) - 1) == date.fromisoformat(headline["window_end"])
    assert round(average["months"][-1], 1) == headline["months"]
    assert average["firm_from"] == "2024-02-02"                      # matched from the first day


def test_payload_records_the_bootstrap_when_draws_are_supplied():
    draws = FakeDraws({("O1", "C1"): 0.5, ("O2", "C1"): 0.9, ("O2", "C2"): 0.2, ("O2", "C3"): 0.0,
                       ("O1", "C2"): 0.0, ("O1", "C3"): 0.0})
    payload = build_payload(payload_models(), date(2025, 6, 30), draws)
    assert payload["test"] == {"paired_bootstrap": True, "resamples": 500, "min_share": 0.05}
    leaders = payload["comparisons"]["open_closed"]["leader_frontier"]
    assert [c["test"] for c in leaders] == ["bootstrap", "bootstrap", None]


# ------------------------------------------------------ the two comparisons

def test_china_us_groups_by_country_whatever_the_licence():
    models = [
        closed("US closed", "2024-01-01", 130),
        opened("US open", "2024-03-01", 135, country=UNITED_STATES),        # leads the U.S. side
        closed("CN closed", "2024-04-01", 134.5, country=CHINA),            # the best Chinese model
        opened("CN open", "2024-02-01", 120),
        opened("FR open", "2024-05-01", 150, country="France"),             # neither group
        closed("Nowhere", "2024-05-02", 160, country=""),                   # no country label
    ]
    comparisons = build_payload(models, date(2024, 12, 31))["comparisons"]

    china_us = comparisons["china_us"]
    assert [m["model"] for m in china_us["leader_frontier"]] == ["US closed", "US open"]
    assert [m["model"] for m in china_us["laggard_frontier"]] == ["CN open", "CN closed"]
    assert china_us["models_scored"] == {"laggard": 2, "leader": 2}
    assert china_us["today"]["laggard_model"] == "CN closed"
    assert china_us["today"]["reference_model"] == "US open"                # within a point of it

    open_closed = comparisons["open_closed"]
    assert open_closed["models_scored"] == {"laggard": 3, "leader": 3}
    assert open_closed["today"]["laggard_model"] == "FR open"
    assert open_closed["today"]["leader_model"] == "Nowhere"


# --------------------------------------------------------------- trendline

def on_line(prefix, is_open, start_eci, per_day, days, country):
    """Models lying exactly on a straight line from January 1, 2024."""
    origin = date(2024, 1, 1)
    return [Model(f"{prefix}{d}", origin + timedelta(days=d), start_eci + per_day * d, is_open, country=country)
            for d in days]


def test_trend_lag_is_the_horizontal_distance_between_the_two_fitted_lines():
    leaders = on_line("us", False, 100, 0.10, (0, 100, 250, 400, 600), UNITED_STATES)
    laggards = on_line("cn", True, 90, 0.08, (30, 200, 380, 500), CHINA)
    trend = trend_lag(laggards, leaders)

    # Read on day 500, the latest Chinese record: its line is at 90 + 0.08 * 500 = 130,
    # a level the U.S. line (100 + 0.10 * day) reached on day 300.
    assert trend["on"] == "2025-05-15" and trend["laggard_model"] == "cn500"
    assert trend["level"] == pytest.approx(130.0)
    assert trend["leader_reached"] == (date(2024, 1, 1) + timedelta(days=300)).isoformat()
    assert trend["lag_months"] == pytest.approx(200 / DAYS_PER_MONTH, abs=0.005)
    # The U.S. line is steeper, so the gap widens: 1 - 0.08/0.10 of a month per month.
    assert trend["change_months_per_year"] == pytest.approx(2.4)
    assert trend["since"] == "2024-01-01"
    assert trend["leader"] == {"records": 5, "eci_per_month": round(0.10 * DAYS_PER_MONTH, 2), "r2": 1.0,
                               "line": [["2024-01-01", 100.0], ["2025-08-23", 160.0]]}
    assert trend["laggard"] == {"records": 4, "eci_per_month": round(0.08 * DAYS_PER_MONTH, 2), "r2": 1.0,
                                "line": [["2024-01-31", 92.4], ["2025-08-23", 138.0]]}


def test_trend_lag_fits_true_records_only():
    leaders = on_line("us", False, 100, 0.10, (100, 250, 400, 600), UNITED_STATES)
    laggards = on_line("cn", True, 90, 0.08, (30, 200, 380, 500), CHINA)
    baseline = trend_lag(laggards, leaders)
    extras = [
        # The U.S. record standing when 2024 begins: too early to be fitted...
        Model("us-2023", date(2023, 11, 6), 108.0, False, country=UNITED_STATES),
        # ...but it stops the first U.S. release of 2024 from counting as a record.
        Model("us-small", date(2024, 1, 25), 60.0, False, country=UNITED_STATES),
        # Not records either: below the best so far, or second best on its day.
        Model("us-behind", date(2024, 9, 1), 105.0, False, country=UNITED_STATES),
        Model("cn-same-day", date(2024, 1, 31), 80.0, True, country=CHINA),
    ]
    assert trend_lag(laggards + extras[3:], leaders + extras[:3]) == baseline
    assert baseline["leader"]["records"] == 4


def test_trend_lag_needs_three_records_a_side():
    leaders = on_line("us", False, 100, 0.10, (0, 100, 250), UNITED_STATES)
    laggards = on_line("cn", True, 90, 0.08, (30, 200, 380), CHINA)
    assert trend_lag(laggards, leaders) is not None
    assert trend_lag(laggards[:2], leaders) is None
    assert trend_lag(laggards, leaders[:2]) is None
    # Records from before the cut-off do not count toward the three.
    early = [Model("cn-early", date(2023, 6, 1), 50.0, True, country=CHINA)]
    assert trend_lag(early + laggards[:2], leaders) is None


def test_trend_lag_is_negative_when_the_laggards_line_is_above():
    leaders = on_line("us", False, 100, 0.10, (0, 100, 250), UNITED_STATES)
    laggards = on_line("cn", True, 120, 0.10, (30, 200, 380), CHINA)
    trend = trend_lag(laggards, leaders)
    assert trend["lag_months"] == pytest.approx(-200 / DAYS_PER_MONTH, abs=0.005)   # 20 points up, 0.10 a day
    assert trend["change_months_per_year"] == 0.0


def test_only_the_country_comparison_publishes_a_trendline():
    models = (on_line("us", False, 100, 0.10, (0, 100, 250, 400, 600), UNITED_STATES)
              + on_line("cn", True, 90, 0.08, (30, 200, 380, 500), CHINA))
    comparisons = build_payload(models, date(2026, 1, 1))["comparisons"]
    assert comparisons["open_closed"]["trend"] is None
    assert comparisons["china_us"]["trend"] == trend_lag(models[5:], models[:5])
    assert json.loads(json.dumps(comparisons["china_us"]["trend"]))["lag_months"] == round(200 / DAYS_PER_MONTH, 2)


def test_a_comparison_with_an_empty_side_is_an_error():
    with pytest.raises(ValueError, match="each group"):
        build_payload([closed("US", "2024-01-01", 130), opened("FR", "2024-02-01", 120, country="France")],
                      date(2024, 12, 31))
    with pytest.raises(ValueError, match="each group"):
        build_comparison([], [closed("US", "2024-01-01", 130)], date(2024, 12, 31), Criterion())
