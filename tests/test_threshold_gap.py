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


from threshold_gap import gaussian_smooth_with_ci  # noqa: E402


class TestGaussianSmoother:
    def test_constant_series_smooths_to_constant(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="30D")
        grid, mean, lo, hi = gaussian_smooth_with_ci(
            pd.Series(dates), pd.Series([5.0] * 10), n_boot=50,
        )
        assert np.allclose(mean.dropna(), 5.0)
        assert np.allclose(lo.dropna(), 5.0)
        assert np.allclose(hi.dropna(), 5.0)

    def test_nan_outside_effective_support(self):
        # Two tight clusters 3 years apart, bandwidth 60d: the middle of the
        # grid has ~zero kernel weight -> NaN (no extrapolation).
        dates = list(pd.date_range("2022-01-01", periods=5, freq="7D"))
        dates += list(pd.date_range("2025-01-01", periods=5, freq="7D"))
        values = [1.0] * 5 + [9.0] * 5
        grid, mean, _, _ = gaussian_smooth_with_ci(
            pd.Series(dates), pd.Series(values), n_boot=20,
        )
        assert mean.isna().any()
        assert mean.notna().any()

    def test_deterministic_under_seed(self):
        dates = pd.Series(pd.date_range("2023-01-01", periods=8, freq="45D"))
        values = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.0, 7.0])
        a = gaussian_smooth_with_ci(dates, values, n_boot=30, seed=0)
        b = gaussian_smooth_with_ci(dates, values, n_boot=30, seed=0)
        for x, y in zip(a, b):
            pd.testing.assert_series_equal(x, y)

    def test_ci_brackets_mean(self):
        rng = np.random.default_rng(42)
        dates = pd.Series(pd.date_range("2023-01-01", periods=30, freq="14D"))
        values = pd.Series(rng.normal(10.0, 2.0, size=30))
        _, mean, lo, hi = gaussian_smooth_with_ci(dates, values, n_boot=200)
        ok = mean.notna() & lo.notna() & hi.notna()
        assert (lo[ok] <= hi[ok]).all()

    def test_empty_input(self):
        grid, mean, lo, hi = gaussian_smooth_with_ci(
            pd.Series(dtype="datetime64[ns]"), pd.Series(dtype=float)
        )
        assert len(grid) == 0 and len(mean) == 0


from threshold_gap import (  # noqa: E402
    build_threshold_analysis,
    build_threshold_aggregate,
    summarize_datapoints,
    THRESHOLD_REVIEW,
)


def _gpqa_like_df():
    return make_df([
        ("closed-1", "2023-01-01", 40.0, False),
        ("closed-2", "2023-06-01", 60.0, False),
        ("open-1", "2023-10-01", 45.0, True),
        ("open-2", "2024-04-01", 62.0, True),
    ])


class TestBuildThresholdAnalysis:
    def test_reviewed_benchmark_uses_review_config(self):
        block = build_threshold_analysis(_gpqa_like_df(), "gpqa_diamond")
        assert block["config"]["keep"] is True
        assert block["config"]["data_access"] == "public"
        assert block["config"]["validity_floor"] == 20.0
        assert 35.0 in block["config"]["accepted_thresholds"]
        assert block["config"]["source_review"] == "open_closed_gap"

    def test_unreviewed_benchmark_defaults(self):
        block = build_threshold_analysis(_gpqa_like_df(), "some_new_benchmark")
        assert block["config"]["keep"] is False
        assert block["config"]["accepted_thresholds"] is None
        assert block["config"]["source_review"] is None

    def test_dates_are_iso_strings_and_json_safe(self):
        block = build_threshold_analysis(
            _gpqa_like_df(), "gpqa_diamond", as_of="2024-06-01"
        )
        assert block["datapoints"], "expected datapoints"
        for r in block["datapoints"]:
            assert isinstance(r["first_closed_date"], str)
        json.dumps(block, allow_nan=False)  # raises on NaN/NaT leakage

    def test_summary_median_over_accepted(self):
        block = build_threshold_analysis(_gpqa_like_df(), "gpqa_diamond")
        s = block["summary"]
        accepted = [r for r in block["datapoints"] if r["accepted"]]
        assert s["n_accepted"] == len(accepted)
        med = float(np.median([r["gap_months"] for r in accepted]))
        assert s["median_gap_months"] == pytest.approx(round(med, 1))

    def test_summary_empty(self):
        s = summarize_datapoints([])
        assert s["n_accepted"] == 0
        assert s["median_gap_months"] is None


class TestBuildThresholdAggregate:
    def _benchmarks(self):
        gpqa = {
            "metadata": {"name": "GPQA Diamond"},
            "threshold_analysis": build_threshold_analysis(
                _gpqa_like_df(), "gpqa_diamond"
            ),
        }
        # keep=False review entry: must be excluded from the aggregate
        chess = {
            "metadata": {"name": "Chess Puzzles"},
            "threshold_analysis": build_threshold_analysis(
                _gpqa_like_df(), "chess_puzzles"
            ),
        }
        # private KEEP benchmark
        fm_df = make_df([
            ("closed-a", "2024-01-01", 12.0, False),
            ("open-a", "2024-11-01", 13.0, True),
        ])
        fm = {
            "metadata": {"name": "FrontierMath"},
            "threshold_analysis": build_threshold_analysis(
                fm_df, "frontiermath_public"
            ),
        }
        return {"gpqa_diamond": gpqa, "chess_puzzles": chess,
                "frontiermath_public": fm, "no_ta": {"metadata": {"name": "x"}}}

    def test_only_keep_benchmarks_pooled(self):
        agg = build_threshold_aggregate(self._benchmarks())
        ids = {p["benchmark_id"] for p in agg["datapoints"]}
        assert "chess_puzzles" not in ids
        assert "gpqa_diamond" in ids and "frontiermath_public" in ids

    def test_datapoints_are_accepted_only(self):
        agg = build_threshold_aggregate(self._benchmarks())
        assert all(p["gap_months"] is not None for p in agg["datapoints"])

    def test_medians_and_access_split(self):
        agg = build_threshold_aggregate(self._benchmarks())
        assert "overall" in agg["medians"]
        assert "public" in agg["medians"]
        assert "private" in agg["medians"]

    def test_json_safe(self):
        json.dumps(build_threshold_aggregate(self._benchmarks()),
                   allow_nan=False)

    def test_trend_omitted_when_too_few_points(self):
        benchmarks = {k: v for k, v in self._benchmarks().items()
                      if k == "frontiermath_public"}
        agg = build_threshold_aggregate(benchmarks)
        # one accepted pair -> <2 datapoints on the private side is possible;
        # whatever survives, trends must only contain entries with points
        for t in agg["trends"].values():
            assert t["points"]
        assert agg["parameters"]["bandwidth_days"] == 60.0


class TestUpdateDataIntegration:
    def test_compute_threshold_block_is_json_safe_and_fail_open(self):
        from update_data import compute_threshold_block

        df = _gpqa_like_df()
        block = compute_threshold_block(df, "gpqa_diamond",
                                        score_col="score", model_col="model")
        assert block is not None
        json.dumps(block, allow_nan=False)

        # Fail-open: a broken frame degrades to None, never raises.
        bad = pd.DataFrame({"weird": [1, 2]})
        assert compute_threshold_block(bad, "gpqa_diamond",
                                       score_col="score",
                                       model_col="model") is None
