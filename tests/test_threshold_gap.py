"""Tests for the threshold-crossing gap analysis (ported from open_closed_gap).

Run with: pytest tests/test_threshold_gap.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from threshold_gap import (
    auto_thresholds,
    auto_validity_floor,
    compute_threshold_datapoints,
    DAYS_PER_MONTH,
)


def make_df(rows):
    """rows: list of (model, date, score, is_open)."""
    return pd.DataFrame(
        [
            {"model": m, "date": pd.Timestamp(d), "score": s, "Open": o}
            for m, d, s, o in rows
        ]
    )


class TestAutoThresholds:
    def test_fraction_scale(self):
        t = auto_thresholds(pd.Series([0.1, 0.5, 0.9]))
        assert t[0] == pytest.approx(0.05)
        assert t[-1] == pytest.approx(0.95)
        assert np.allclose(np.diff(t), 0.05)

    def test_percentage_scale(self):
        t = auto_thresholds(pd.Series([10.0, 50.0, 92.0]))
        assert t[0] == pytest.approx(5.0)
        assert t[-1] == pytest.approx(95.0)
        assert np.allclose(np.diff(t), 5.0)

    def test_large_scale_linspace(self):
        t = auto_thresholds(pd.Series([100.0, 150.0]))
        assert len(t) == 19  # interior of 21-point linspace
        assert t.min() > 100.0 and t.max() < 150.0

    def test_validity_floor_is_10th_percentile(self):
        scores = pd.Series(np.arange(1.0, 101.0))
        assert auto_validity_floor(scores) == pytest.approx(
            np.percentile(scores, 10)
        )


class TestComputeDatapoints:
    def test_first_crosser_is_earliest_not_highest(self):
        df = make_df([
            ("closed-early-low", "2023-01-01", 55.0, False),
            ("closed-late-high", "2023-06-01", 90.0, False),
            ("open-a", "2024-01-01", 60.0, True),
        ])
        rows = compute_threshold_datapoints(df, thresholds=[50.0])
        assert len(rows) == 1
        assert rows[0]["first_closed_model"] == "closed-early-low"
        assert rows[0]["first_open_model"] == "open-a"
        assert rows[0]["gap_days"] == 365

    def test_gap_months_uses_days_per_month(self):
        df = make_df([
            ("c", "2023-01-01", 80.0, False),
            ("o", "2023-12-31", 80.0, True),
        ])
        rows = compute_threshold_datapoints(df, thresholds=[75.0])
        assert rows[0]["gap_months"] == pytest.approx(
            round(364 / DAYS_PER_MONTH, 1)
        )

    def test_negative_gap_when_open_led(self):
        df = make_df([
            ("o", "2023-01-01", 80.0, True),
            ("c", "2023-03-01", 80.0, False),
        ])
        rows = compute_threshold_datapoints(df, thresholds=[75.0])
        assert rows[0]["gap_days"] < 0

    def test_threshold_skipped_when_no_closed_crosser(self):
        df = make_df([
            ("c", "2023-01-01", 40.0, False),
            ("o", "2023-06-01", 80.0, True),
        ])
        rows = compute_threshold_datapoints(df, thresholds=[50.0])
        assert rows == []

    def test_still_open_when_no_open_crosser(self):
        df = make_df([
            ("c", "2023-01-01", 80.0, False),
            ("o", "2023-06-01", 40.0, True),
        ])
        rows = compute_threshold_datapoints(
            df, thresholds=[75.0], as_of="2023-07-01"
        )
        assert len(rows) == 1
        r = rows[0]
        assert r["still_open"] is True
        assert r["gap_days"] is None and r["gap_months"] is None
        assert r["accepted"] is False and r["valid"] is False
        # 181 days from 2023-01-01 to 2023-07-01
        assert r["gap_months_so_far"] == pytest.approx(
            round(181 / DAYS_PER_MONTH, 1)
        )

    def test_empty_when_one_side_missing(self):
        only_closed = make_df([("c", "2023-01-01", 80.0, False)])
        assert compute_threshold_datapoints(only_closed, thresholds=[50.0]) == []
        only_open = make_df([("o", "2023-01-01", 80.0, True)])
        assert compute_threshold_datapoints(only_open, thresholds=[50.0]) == []

    def test_nan_rows_dropped(self):
        df = make_df([
            ("c", "2023-01-01", 80.0, False),
            ("o", "2023-06-01", 80.0, True),
        ])
        df.loc[len(df)] = {"model": "bad", "date": pd.NaT,
                           "score": 99.0, "Open": False}
        rows = compute_threshold_datapoints(df, thresholds=[75.0])
        assert rows[0]["first_closed_model"] == "c"


class TestValidityAndDedup:
    def _pair_df(self):
        # closed C1 crosses 50/55/60 first at 2023-01-01 (score 62);
        # open O1 crosses the same three at 2023-07-01 (score 63)
        # => one pair spanning thresholds 50, 55, 60.
        return make_df([
            ("C1", "2023-01-01", 62.0, False),
            ("O1", "2023-07-01", 63.0, True),
        ])

    def test_one_accepted_representative_at_highest_threshold(self):
        rows = compute_threshold_datapoints(
            self._pair_df(), thresholds=[50.0, 55.0, 60.0],
            accepted_thresholds=[50.0, 55.0, 60.0],
        )
        accepted = [r for r in rows if r["accepted"]]
        assert len(accepted) == 1
        assert accepted[0]["threshold"] == 60.0
        assert all(r["valid"] for r in rows)

    def test_representative_can_sit_outside_allowlist(self):
        # Only T=50 is review-accepted, but the pair spans up to 60:
        # the representative datapoint moves to the pair's highest threshold.
        rows = compute_threshold_datapoints(
            self._pair_df(), thresholds=[50.0, 55.0, 60.0],
            accepted_thresholds=[50.0],
        )
        accepted = [r for r in rows if r["accepted"]]
        assert len(accepted) == 1
        assert accepted[0]["threshold"] == 60.0
        assert accepted[0]["valid"] is False  # not itself allowlisted

    def test_allowlist_overrides_floor(self):
        rows = compute_threshold_datapoints(
            self._pair_df(), thresholds=[50.0, 55.0, 60.0],
            accepted_thresholds=[55.0], validity_floor=0.0,
        )
        by_t = {r["threshold"]: r for r in rows}
        assert by_t[50.0]["valid"] is False
        assert by_t[55.0]["valid"] is True
        assert by_t[60.0]["valid"] is False

    def test_floor_validity_is_strictly_above(self):
        rows = compute_threshold_datapoints(
            self._pair_df(), thresholds=[50.0, 55.0], validity_floor=50.0,
        )
        by_t = {r["threshold"]: r for r in rows}
        assert by_t[50.0]["valid"] is False  # t > floor required
        assert by_t[55.0]["valid"] is True

    def test_allowlist_membership_is_float_rounding_safe(self):
        rows = compute_threshold_datapoints(
            self._pair_df(), thresholds=[0.1 + 0.2],  # 0.30000000000000004
            accepted_thresholds=[0.3],
        )
        assert rows[0]["valid"] is True

    def test_two_distinct_pairs_two_representatives(self):
        df = make_df([
            ("C1", "2023-01-01", 55.0, False),   # first closed over 50
            ("C2", "2023-03-01", 82.0, False),   # first closed over 80
            ("O1", "2023-09-01", 57.0, True),    # first open over 50
            ("O2", "2024-01-01", 85.0, True),    # first open over 80
        ])
        rows = compute_threshold_datapoints(
            df, thresholds=[50.0, 80.0],
            accepted_thresholds=[50.0, 80.0],
        )
        accepted = [r for r in rows if r["accepted"]]
        assert len(accepted) == 2
        assert {(r["first_closed_model"], r["first_open_model"])
                for r in accepted} == {("C1", "O1"), ("C2", "O2")}
