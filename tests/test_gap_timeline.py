"""Tests for scripts/gap_timeline.py (gap-by-date timeline + context stats).

Run with: pytest tests/test_gap_timeline.py -v
"""

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from scripts.gap_timeline import DAYS_PER_MONTH, build_gap_timeline, build_score_gap_timeline
from update_data import _open_caught_up, calculate_gap_metrics, calculate_horizontal_gaps


def make_df(rows, std=None):
    """rows: (model, date, score, Open). std: optional list of score_std."""
    df = pd.DataFrame(rows, columns=["model", "date", "score", "Open"])
    df["date"] = pd.to_datetime(df["date"])
    if std is not None:
        df["score_std"] = std
    return df


def threshold_caught_up(open_score, open_std, sota_score, sota_std,
                        open_name=None, sota_name=None, threshold=1.0):
    return open_score >= sota_score - threshold


def months(a, b):
    return (pd.Timestamp(a) - pd.Timestamp(b)).days / DAYS_PER_MONTH


def build(df, as_of, **kw):
    return build_gap_timeline(
        df, score_col="score", model_col="model",
        caught_up=threshold_caught_up, as_of=as_of, **kw)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------

def test_empty_df_returns_none():
    df = make_df([])
    assert build(df, "2025-01-01") is None


def test_one_sided_returns_none():
    only_leader = make_df([("A", "2024-01-01", 100.0, False)])
    only_laggard = make_df([("B", "2024-01-01", 100.0, True)])
    assert build(only_leader, "2025-01-01") is None
    assert build(only_laggard, "2025-01-01") is None


def test_as_of_before_window_returns_none():
    df = make_df([("A", "2024-01-01", 100.0, False), ("B", "2024-04-01", 100.5, True)])
    # window starts at the first laggard release (Apr 1); as_of is earlier.
    assert build(df, "2024-03-01") is None


def test_models_after_as_of_are_ignored():
    df = make_df([
        ("A", "2024-01-01", 100.0, False),
        ("B", "2024-04-01", 100.5, True),
        ("C", "2024-07-01", 110.0, False),   # after as_of
        ("D", "2024-10-01", 110.5, True),    # after as_of
    ])
    out = build(df, "2024-06-01")
    assert out is not None
    assert len(out["segments"]) == 1
    assert out["segments"][0]["reference_model"] == "A"
    assert out["events"] == []


# ---------------------------------------------------------------------------
# Single rising segment
# ---------------------------------------------------------------------------

def test_single_segment_geometry():
    df = make_df([("A", "2024-01-01", 100.0, False), ("B", "2024-04-01", 100.5, True)])
    out = build(df, "2024-07-01")
    assert out["start"] == "2024-04-01T00:00:00"
    assert out["as_of"] == "2024-07-01T00:00:00"
    assert out["n_days"] == (pd.Timestamp("2024-07-01") - pd.Timestamp("2024-04-01")).days + 1
    (seg,) = out["segments"]
    assert seg["start"] == "2024-04-01T00:00:00"
    assert seg["end"] == "2024-07-01T00:00:00"
    assert seg["laggard_model"] == "B"
    assert seg["reference_model"] == "A"
    assert seg["reference_matched"] is True
    assert seg["gap_start"] == pytest.approx(round(months("2024-04-01", "2024-01-01"), 1))
    assert seg["gap_end"] == pytest.approx(round(months("2024-07-01", "2024-01-01"), 1))
    ctx = out["context"]
    assert ctx["current_gap_months"] == seg["gap_end"]
    assert ctx["current_reference_model"] == "A"
    assert ctx["current_laggard_model"] == "B"
    assert ctx["current_since"] == "2024-04-01T00:00:00"
    # Monotonically rising from the start: a record high, and never smaller
    # before the current run (which is the whole history).
    assert ctx["is_record_high"] is True
    assert ctx["last_at_least"]["date"] is None
    assert ctx["last_at_least"]["run_since"] is None
    assert ctx["last_at_most"]["date"] is None
    assert ctx["last_at_most"]["run_since"] == "2024-04-01T00:00:00"
    assert ctx["is_record_low"] is False
    assert ctx["peak"]["date"] == "2024-07-01T00:00:00"
    assert ctx["trough"]["date"] == "2024-04-01T00:00:00"
    assert ctx["percentile_of_current"] == 100
    assert ctx["year_ago"] is None


# ---------------------------------------------------------------------------
# Sawtooth: laggard release matching a newer leader SOTA drops the gap
# ---------------------------------------------------------------------------

SAWTOOTH = [
    ("A", "2024-01-01", 100.0, False),
    ("B", "2024-04-01", 100.5, True),   # matches A
    ("C", "2024-07-01", 110.0, False),  # B does not match C -> no change
    ("D", "2024-10-01", 110.5, True),   # matches C -> gap drops
]


def test_sawtooth_segments_and_event():
    out = build(make_df(SAWTOOTH), "2025-01-01")
    segs = out["segments"]
    # Segments split on (laggard, reference, next unmatched leader): C's
    # release on Jul 1 opens a new segment even though the matched lead is
    # continuous across it.
    assert [(s["laggard_model"], s["reference_model"]) for s in segs] == [("B", "A"), ("B", "A"), ("D", "C")]
    s1, s2, s3 = segs
    assert s1["end"] == "2024-07-01T00:00:00"
    assert s1["gap_end"] == s2["gap_start"]
    assert s2["end"] == "2024-10-01T00:00:00"
    assert s2["gap_end"] == pytest.approx(round(months("2024-10-01", "2024-01-01"), 1))  # 9.0
    assert s3["start"] == "2024-10-01T00:00:00"
    assert s3["gap_start"] == pytest.approx(round(months("2024-10-01", "2024-07-01"), 1))  # 3.0
    assert s3["end"] == "2025-01-01T00:00:00"
    (ev,) = out["events"]
    assert ev["date"] == "2024-10-01T00:00:00"
    assert ev["kind"] == "laggard_release"
    assert ev["model"] == "D"
    assert ev["gap_before"] == s2["gap_end"]
    assert ev["gap_after"] == s3["gap_start"]
    assert ev["laggard_model"] == "D"
    assert ev["reference_model"] == "C"


def test_last_model_date_counts_non_frontier_releases_before_as_of():
    rows = SAWTOOTH + [
        ("X", "2024-11-15", 90.0, False),   # below the frontier, but scored
        ("Z", "2025-06-01", 130.0, False),  # after as_of: ignored
    ]
    out = build(make_df(rows), "2025-01-01")
    assert out["last_model_date"] == "2024-11-15T00:00:00"
    assert out["as_of"] == "2025-01-01T00:00:00"


def test_frontier_lists_are_running_max_sequences():
    rows = SAWTOOTH + [
        ("X", "2024-08-01", 105.0, False),   # below C: not a leader record
        ("Y", "2024-11-01", 108.0, True),    # below D: not a laggard record
        ("Z", "2024-12-01", 130.0, False),   # unmatched leader record
    ]
    out = build(make_df(rows), "2025-01-01")
    assert [r["model"] for r in out["leader_frontier"]] == ["A", "C", "Z"]
    assert [r["model"] for r in out["laggard_frontier"]] == ["B", "D"]
    assert out["leader_frontier"][2] == {"model": "Z", "date": "2024-12-01T00:00:00", "score": 130.0}


def test_sawtooth_context_lookbacks():
    out = build(make_df(SAWTOOTH), "2025-01-01")
    ctx = out["context"]
    cur_raw = months("2025-01-01", "2024-07-01")  # 184 days
    assert ctx["current_gap_months"] == pytest.approx(round(cur_raw, 1))
    assert ctx["current_since"] == "2024-10-01T00:00:00"
    # Rising tooth: no current >=cur run; the previous time the gap was at
    # least this large was the last day of the first tooth (Sep 30).
    al = ctx["last_at_least"]
    assert al["run_since"] is None
    assert al["date"] == "2024-09-30T00:00:00"
    assert al["gap_months"] == pytest.approx(round(months("2024-09-30", "2024-01-01"), 1))
    assert al["months_ago"] == pytest.approx(round(months("2025-01-01", "2024-09-30"), 1))
    assert ctx["is_record_high"] is False
    # <=cur run is the whole current tooth (since Oct 1); before that, the
    # last day at or below cur is the day in tooth 1 with exactly 184 days
    # since A (Jul 3, 2024) -- ties count as "this small".
    am = ctx["last_at_most"]
    assert am["run_since"] == "2024-10-01T00:00:00"
    assert am["date"] == "2024-07-03T00:00:00"
    assert am["gap_months"] == pytest.approx(round(cur_raw, 1))
    assert ctx["is_record_low"] is False
    # Peak = day before the drop; trough = first day (2.99 < 3.02).
    assert ctx["peak"]["date"] == "2024-09-30T00:00:00"
    assert ctx["peak"]["laggard_model"] == "B"
    assert ctx["peak"]["reference_model"] == "A"
    assert ctx["trough"]["date"] == "2024-04-01T00:00:00"
    assert ctx["trough"]["gap_months"] == pytest.approx(round(months("2024-04-01", "2024-01-01"), 1))
    assert 0 < ctx["percentile_of_current"] < 100
    assert ctx["year_ago"] is None  # 2024-01-01 is before the window start


def test_dropped_today_reports_ongoing_run():
    """as_of == a drop date: the gap was >= today's value continuously since
    the day tooth 1 crossed today's level, and never before that."""
    out = build(make_df(SAWTOOTH), "2024-10-01")
    ctx = out["context"]
    cur_raw = months("2024-10-01", "2024-07-01")  # 92 days
    assert ctx["current_gap_months"] == pytest.approx(round(cur_raw, 1))
    al = ctx["last_at_least"]
    # tooth 1 reached 92 days-since-A on Apr 2 (tie counts as >=)
    assert al["run_since"] == "2024-04-02T00:00:00"
    assert al["date"] is None
    assert ctx["is_record_high"] is False  # yesterday was larger
    am = ctx["last_at_most"]
    assert am["run_since"] is None  # today starts a fresh <=cur run
    # Apr 2 was exactly 92 days since A: a tie counts as "at most this small".
    assert am["date"] == "2024-04-02T00:00:00"
    assert ctx["is_record_low"] is False


def test_year_ago_change():
    rows = SAWTOOTH + [("E", "2025-04-01", 120.0, False), ("F", "2025-07-01", 120.5, True)]
    out = build(make_df(rows), "2025-10-01")
    ctx = out["context"]
    ya = ctx["year_ago"]
    assert ya["date"] == "2024-10-01T00:00:00"
    assert ya["gap_months"] == pytest.approx(round(months("2024-10-01", "2024-07-01"), 1))
    cur = months("2025-10-01", "2025-04-01")
    assert ctx["current_gap_months"] == pytest.approx(round(cur, 1))
    assert ya["change_months"] == pytest.approx(round(cur - months("2024-10-01", "2024-07-01"), 1))


def test_trailing_12m_means():
    rows = SAWTOOTH + [("E", "2025-04-01", 120.0, False), ("F", "2025-07-01", 120.5, True)]
    df = make_df(rows)
    # < 365 days of data: no trailing block at all.
    assert build(df, "2025-03-01")["context"]["trailing_12m"] is None
    # >= 365 but < 730 days: trailing mean only.
    t1 = build(df, "2025-05-01")["context"]["trailing_12m"]
    assert t1["mean_gap_months"] is not None
    assert t1["prior_mean_gap_months"] is None and t1["change_months"] is None
    # >= 730 days: both windows, and the change is their difference.
    out = build(df, "2026-05-01")
    t2 = out["context"]["trailing_12m"]
    # Recompute from the published segments.
    _, _, vals = _daily_values_from_segments(out)
    exp_t = float(np.mean(vals[-365:]))
    exp_p = float(np.mean(vals[-730:-365]))
    assert t2["mean_gap_months"] == pytest.approx(round(exp_t, 1))
    assert t2["prior_mean_gap_months"] == pytest.approx(round(exp_p, 1))
    assert t2["change_months"] == pytest.approx(round(exp_t - exp_p, 1))


def test_zero_gap_today_is_record_low_with_null_lookbacks():
    """cur == 0 right after a leader release the laggard had already matched:
    no earlier day was <= 0, so last_at_most is empty and is_record_low is
    True -- the flags, not the nulls, carry the record semantics."""
    df = make_df([
        ("A", "2024-01-01", 100.0, False),
        ("B", "2024-04-01", 100.5, True),
        ("C", "2024-07-01", 101.0, False),
    ])
    ctx = build(df, "2024-07-01")["context"]
    assert ctx["current_gap_months"] == 0.0
    assert ctx["is_record_low"] is True
    assert ctx["is_record_high"] is False
    assert ctx["last_at_most"] == {"date": None, "gap_months": None, "months_ago": None, "run_since": None}
    # Every earlier day was >= 0: the >=cur run spans the whole history.
    assert ctx["last_at_least"]["run_since"] == "2024-04-01T00:00:00"
    assert ctx["last_at_least"]["date"] is None


def test_reversal_when_new_laggard_record_fails_a_match_the_old_one_passed():
    """The caught-up test is not monotone in the point estimate (paired
    bootstrap). Simulate: B2 scores above B1 but fails against C."""
    df = make_df([
        ("A", "2024-01-01", 100.0, False),
        ("C", "2024-07-01", 101.0, False),
        ("B1", "2024-08-01", 100.5, True),   # matches C (and A)
        ("B2", "2024-10-01", 100.8, True),   # higher score, but refused vs C
    ])

    def cu(open_score, open_std, sota_score, sota_std, open_name=None, sota_name=None):
        if open_name == "B2" and sota_name == "C":
            return False
        return open_score >= sota_score - 1.0

    out = build_gap_timeline(df, score_col="score", model_col="model",
                             caught_up=cu, as_of="2024-12-01")
    assert [(s["laggard_model"], s["reference_model"]) for s in out["segments"]] == [("B1", "C"), ("B2", "A")]
    (ev,) = out["events"]
    assert ev["kind"] == "laggard_release"
    assert ev["model"] == "B2"
    assert ev["reversal"] is True
    assert ev["gap_before"] == pytest.approx(round(months("2024-10-01", "2024-07-01"), 1))
    assert ev["gap_after"] == pytest.approx(round(months("2024-10-01", "2024-01-01"), 1))
    # Ordinary drops are not reversals.
    drop = build(make_df(SAWTOOTH), "2025-01-01")["events"][0]
    assert drop["reversal"] is False


def test_same_date_leader_pair_prefers_higher_score_reference():
    df = make_df([
        ("A1", "2024-01-01", 100.0, False),
        ("A2", "2024-01-01", 102.0, False),
        ("B", "2024-04-01", 101.5, True),   # matches both A1 and A2
    ])
    out = build(df, "2024-06-01")
    (seg,) = out["segments"]
    assert seg["reference_model"] == "A2"
    assert seg["gap_start"] == pytest.approx(round(months("2024-04-01", "2024-01-01"), 1))
    assert [r["model"] for r in out["leader_frontier"]] == ["A1", "A2"]


def test_equal_score_laggard_keeps_first_record():
    df = make_df([
        ("A", "2024-01-01", 100.0, False),
        ("B1", "2024-04-01", 100.5, True),
        ("B2", "2024-05-01", 100.5, True),   # ties B1: not a new record
    ])
    out = build(df, "2024-06-01")
    assert len(out["segments"]) == 1
    assert out["segments"][0]["laggard_model"] == "B1"
    assert [r["model"] for r in out["laggard_frontier"]] == ["B1"]


# ---------------------------------------------------------------------------
# Leader release that the laggard had already matched resets the gap to 0
# ---------------------------------------------------------------------------

def test_leader_release_already_matched_resets_to_zero():
    df = make_df([
        ("A", "2024-01-01", 100.0, False),
        ("B", "2024-04-01", 100.5, True),
        ("C", "2024-07-01", 101.0, False),  # within B's threshold reach
    ])
    out = build(df, "2024-09-01")
    (ev,) = out["events"]
    assert ev["kind"] == "leader_release"
    assert ev["model"] == "C"
    assert ev["date"] == "2024-07-01T00:00:00"
    assert ev["gap_before"] == pytest.approx(round(months("2024-07-01", "2024-01-01"), 1))
    assert ev["gap_after"] == 0.0
    ctx = out["context"]
    assert ctx["trough"]["gap_months"] == 0.0
    assert ctx["trough"]["date"] == "2024-07-01T00:00:00"
    assert ctx["current_reference_model"] == "C"


def test_laggard_change_without_reference_change_splits_segment_but_no_event():
    df = make_df([
        ("A", "2024-01-01", 100.0, False),
        ("B", "2024-04-01", 100.5, True),
        ("C", "2024-07-01", 120.0, False),
        ("D", "2024-08-01", 105.0, True),   # better laggard, still only matches A
    ])
    out = build(df, "2024-10-01")
    segs = out["segments"]
    # B alone; B once C (unmatched) is out; D after its release.
    assert [s["laggard_model"] for s in segs] == ["B", "B", "D"]
    assert all(s["reference_model"] == "A" for s in segs)
    assert [s["next_leader_model"] for s in segs] == [None, "C", "C"]
    assert out["events"] == []
    # Continuous across every split.
    for a, b in zip(segs, segs[1:]):
        assert a["gap_end"] == b["gap_start"]


def test_unmatched_floor_uses_earliest_sota_and_flags_it():
    df = make_df([
        ("A", "2024-01-01", 100.0, False),
        ("C", "2024-03-01", 120.0, False),
        ("B", "2024-04-01", 50.0, True),  # far below every leader model
    ])
    out = build(df, "2024-06-01")
    (seg,) = out["segments"]
    assert seg["reference_model"] == "A"
    assert seg["reference_matched"] is False
    assert out["context"]["current_reference_matched"] is False


# ---------------------------------------------------------------------------
# Consistency with the headline day-by-day metric
# ---------------------------------------------------------------------------

def _daily_values_from_segments(out):
    """Expand the published segments back to daily values (what the frontend
    does for hover): (mean, n, values)."""
    vals = []
    segs = out["segments"]
    for i, s in enumerate(segs):
        start = pd.Timestamp(s["start"])
        end = pd.Timestamp(s["end"])
        ref = pd.Timestamp(s["reference_date"])
        last = i == len(segs) - 1
        days = pd.date_range(start, end, inclusive="both" if last else "left")
        vals.extend((d - ref).days / DAYS_PER_MONTH for d in days)
    return float(np.mean(vals)), len(vals), np.array(vals)


def _daily_mean_from_segments(out):
    mean, n, _ = _daily_values_from_segments(out)
    return mean, n


@pytest.mark.parametrize("with_std", [False, True])
def test_matches_calculate_gap_metrics_average(with_std):
    rows = [
        ("L1", "2023-03-01", 100.0, False),
        ("G1", "2023-06-01", 90.0, True),
        ("L2", "2023-09-01", 110.0, False),
        ("G2", "2023-11-15", 100.4, True),
        ("L3", "2024-01-10", 111.0, False),
        ("G3", "2024-03-01", 110.2, True),
        ("L4", "2024-05-05", 125.0, False),
        ("G4", "2024-09-01", 118.0, True),
        ("L5", "2024-10-01", 130.0, False),
        ("G5", "2025-02-01", 126.0, True),
        ("L6", "2025-03-15", 140.0, False),
    ]
    std = [1.0, 2.0, 1.5, 1.0, 0.8, 1.2, 1.0, 1.1, 0.9, 1.3, 1.0] if with_std else None
    df = make_df(rows, std=std)
    df["Model"] = df["model"]  # calculate_gap_metrics reads the name from "Model"
    caught_up = partial(_open_caught_up, threshold=1.0)

    def cu(open_score, open_std, sota_score, sota_std, open_name, sota_name):
        return caught_up(open_score, open_std, sota_score, sota_std,
                         open_name=open_name, sota_name=sota_name)

    as_of = df["date"].max()
    out = build_gap_timeline(df, score_col="score", model_col="model",
                             caught_up=cu, as_of=as_of)
    metrics = calculate_gap_metrics(df, score_col="score", threshold=1.0)
    assert out["start"] == metrics["window_start"]
    assert out["n_days"] == metrics["n_days"]
    mean_from_segments, n = _daily_mean_from_segments(out)
    assert n == metrics["n_days"]
    assert mean_from_segments == pytest.approx(metrics["avg_time_gap_months"], abs=1e-9)
    # The context block's own daily expansion must agree too.
    assert out["context"]["mean_gap_months"] == pytest.approx(
        round(metrics["avg_time_gap_months"], 1), abs=0.051)


# ---------------------------------------------------------------------------
# Lower bound (oldest unmatched leader), expected-lead sampling
# ---------------------------------------------------------------------------

def test_lower_bound_tracks_oldest_unmatched_leader():
    out = build(make_df(SAWTOOTH), "2025-01-01")
    segs = out["segments"]
    # Apr-Jul: B matched A and no newer leader exists -> lower bound 0.
    # Jul-Oct: C is out and unmatched -> lower bound = months since C.
    # Oct-Jan: D matched C, nothing newer -> 0 again.
    assert [(s["laggard_model"], s["reference_model"], s["next_leader_model"]) for s in segs] == [
        ("B", "A", None), ("B", "A", "C"), ("D", "C", None)]
    s2 = segs[1]
    assert s2["start"] == "2024-07-01T00:00:00"
    assert s2["lower_start"] == 0.0
    assert s2["lower_end"] == pytest.approx(round(months("2024-10-01", "2024-07-01"), 1))
    assert segs[0]["lower_end"] == 0.0 and segs[2]["lower_start"] == 0.0
    # The split at Jul 1 is not a discontinuity of the matched lead: no event.
    assert [e["date"][:10] for e in out["events"]] == ["2024-10-01"]
    ctx = out["context"]
    assert ctx["current_lower_months"] == 0.0
    assert ctx["current_next_leader_model"] is None
    # With an unmatched leader today, the bound is its age and it is named.
    out2 = build(make_df(SAWTOOTH), "2024-09-01")
    c2 = out2["context"]
    assert c2["current_next_leader_model"] == "C"
    assert c2["current_lower_months"] == pytest.approx(round(months("2024-09-01", "2024-07-01"), 1))
    assert c2["current_lower_months"] < c2["current_gap_months"]


def test_unmatched_floor_names_earliest_leader_as_next():
    df = make_df([
        ("A", "2024-01-01", 100.0, False),
        ("C", "2024-03-01", 120.0, False),
        ("B", "2024-04-01", 50.0, True),
    ])
    (seg,) = build(df, "2024-06-01")["segments"]
    assert seg["reference_matched"] is False
    assert seg["next_leader_model"] == "A"
    assert seg["lower_start"] == seg["gap_start"]


def test_expected_lead_sampling():
    calls = []

    def expected_fn(t):
        calls.append(t)
        return 2.5 if t.day != 15 else None  # one sample dropped

    out = build_gap_timeline(make_df(SAWTOOTH), score_col="score", model_col="model",
                             caught_up=threshold_caught_up, as_of="2025-01-01",
                             expected_fn=expected_fn, expected_every_days=7)
    pts = out["expected_lead"]["points"]
    assert out["expected_lead"]["method"] == "survival"
    assert calls[0] == pd.Timestamp("2024-04-01")
    assert calls[-1] == pd.Timestamp("2025-01-01")  # as_of always sampled
    assert all(p["months"] == 2.5 for p in pts)
    assert pts[-1]["date"] == "2025-01-01T00:00:00"
    assert len(pts) == len([c for c in calls if c.day != 15])
    # Without a sampler the block is present but empty.
    assert build(make_df(SAWTOOTH), "2025-01-01")["expected_lead"]["points"] == []


def test_calculate_horizontal_gaps_as_of_controls_unmatched_age():
    df = make_df([("A", "2024-01-01", 150.0, False), ("B", "2024-04-01", 100.0, True)])
    df["Model"] = df["model"]
    (g,) = calculate_horizontal_gaps(df, score_col="score", threshold=1.0, model_col="Model",
                                     as_of="2024-07-01")
    assert g["matched"] is False
    assert g["gap_months"] == pytest.approx(round(months("2024-07-01", "2024-01-01"), 1))


# ---------------------------------------------------------------------------
# Score gap block
# ---------------------------------------------------------------------------

def test_score_gap_steps_events_and_context():
    out = build_score_gap_timeline(make_df(SAWTOOTH), score_col="score", model_col="model",
                                   as_of="2025-01-01")
    steps = out["steps"]
    assert [(s["leader_model"], s["laggard_model"], s["gap"]) for s in steps] == [
        ("A", "B", -0.5), ("C", "B", 9.5), ("C", "D", -0.5)]
    assert steps[0]["end"] == "2024-07-01T00:00:00"
    assert steps[-1]["end"] == "2025-01-01T00:00:00"
    ev = out["events"]
    assert [(e["date"][:10], e["kind"], e["model"], e["gap_before"], e["gap_after"]) for e in ev] == [
        ("2024-07-01", "leader_release", "C", -0.5, 9.5),
        ("2024-10-01", "laggard_release", "D", 9.5, -0.5)]
    c = out["context"]
    assert c["current_gap_points"] == -0.5
    assert c["laggard_leads"] is True
    assert c["current_leader_model"] == "C" and c["current_laggard_model"] == "D"
    assert c["peak"] == {"gap_points": 9.5, "date": "2024-07-01T00:00:00", "leader_model": "C", "laggard_model": "B"}
    assert c["trough"]["gap_points"] == -0.5 and c["trough"]["date"] == "2024-04-01T00:00:00"
    # last time at least this large: the whole history is >= -0.5, so the run
    # covers everything and there is no earlier day.
    assert c["last_at_least"]["run_since"] == "2024-04-01T00:00:00"
    assert c["last_at_least"]["date"] is None
    assert c["is_record_low"] is True  # ties with the earlier -0.5 days
    assert 0 < c["percentile_of_current"] < 100


def test_score_gap_last_time_this_large():
    rows = [
        ("A", "2024-01-01", 100.0, False),
        ("B", "2024-02-01", 98.0, True),    # gap 2
        ("C", "2024-04-01", 110.0, False),  # gap 12
        ("D", "2024-06-01", 108.0, True),   # gap 2
        ("E", "2024-09-01", 116.0, False),  # gap 8 today
    ]
    c = build_score_gap_timeline(make_df(rows), score_col="score", model_col="model",
                                 as_of="2024-10-01")["context"]
    assert c["current_gap_points"] == 8.0
    al = c["last_at_least"]
    assert al["run_since"] == "2024-09-01T00:00:00"      # >= 8 since E's release
    assert al["date"] == "2024-05-31T00:00:00"           # last day of the 12-point plateau
    assert al["gap_points"] == 12.0
    assert c["is_record_high"] is False


def test_score_gap_mean_matches_calculate_gap_metrics_when_non_negative():
    rows = [
        ("L1", "2023-03-01", 100.0, False), ("G1", "2023-06-01", 90.0, True),
        ("L2", "2023-09-01", 110.0, False), ("G2", "2023-11-15", 100.4, True),
        ("L3", "2024-01-10", 111.0, False), ("G3", "2024-03-01", 110.2, True),
        ("L4", "2024-05-05", 125.0, False), ("G4", "2024-09-01", 118.0, True),
    ]
    df = make_df(rows)
    df["Model"] = df["model"]
    out = build_score_gap_timeline(df, score_col="score", model_col="model", as_of=df["date"].max())
    metrics = calculate_gap_metrics(df, score_col="score", threshold=1.0)
    assert out["n_days"] == metrics["n_days"]
    assert out["context"]["mean_gap_points"] == pytest.approx(round(metrics["avg_vertical_gap"], 1), abs=0.051)


def test_score_gap_json_serialisable():
    out = build_score_gap_timeline(make_df(SAWTOOTH), score_col="score", model_col="model",
                                   as_of="2025-01-01")
    back = json.loads(json.dumps(out))
    assert back["method"] == "score_gap"
    assert isinstance(back["context"]["percentile_of_current"], int)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_output_is_json_serialisable_and_rounded():
    out = build(make_df(SAWTOOTH), "2025-01-01")
    text = json.dumps(out)
    back = json.loads(text)
    assert back["method"] == "epoch_daily"
    for s in back["segments"]:
        for k in ("gap_start", "gap_end"):
            assert round(s[k], 1) == s[k]
        for k in ("laggard_score", "reference_score"):
            assert round(s[k], 2) == s[k]
    ctx = back["context"]
    assert isinstance(ctx["percentile_of_current"], int)
    assert isinstance(back["n_days"], int)
