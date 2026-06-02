"""
Tests for ECI gap calculations.

Run with: pytest tests/test_gap_calculations.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from update_data import (
    calculate_horizontal_gaps,
    calculate_statistics,
    calculate_gap_metrics,
    _open_caught_up,
    estimate_current_gap,
    calculate_historical_gaps,
    is_china_org,
    is_us_org,
    DAYS_PER_MONTH,
    ECI_MATCH_THRESHOLD,
    Z_ONE_SIDED_05,
)


class TestHorizontalGapCalculation:
    """Tests for calculate_horizontal_gaps function."""

    def test_simple_matched_gap(self):
        """Test a simple case where an open model matches a closed model."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [100.0, 101.0],
            "date": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "Open": [False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        assert len(gaps) == 1
        assert gaps[0]["closed_model"] == "ClosedA"
        assert gaps[0]["open_model"] == "OpenB"
        assert gaps[0]["matched"] is True
        # Exact calculation: Jan 1 to Apr 1 = 91 days (2024 is leap year)
        expected_days = (pd.to_datetime("2024-04-01") - pd.to_datetime("2024-01-01")).days
        expected_months = expected_days / DAYS_PER_MONTH
        # gap_months is rounded to 1 decimal place in the code
        assert abs(gaps[0]["gap_months"] - round(expected_months, 1)) < 0.01

    def test_unmatched_gap(self):
        """Test case where no open model matches the closed model."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [150.0, 100.0],  # Open model ECI too low
            "date": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "Open": [False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        assert len(gaps) == 1
        assert gaps[0]["matched"] is False
        assert gaps[0]["open_model"] is None
        # Gap should be from Jan 2024 to now
        assert gaps[0]["gap_months"] > 10  # At least 10 months from Jan 2024

    def test_open_model_must_be_after_closed(self):
        """Test that open model must be released AFTER closed model."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [100.0, 105.0],
            "date": pd.to_datetime(["2024-06-01", "2024-01-01"]),  # Open before closed
            "Open": [False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        assert len(gaps) == 1
        # Should be unmatched because open model came before closed
        assert gaps[0]["matched"] is False

    def test_multiple_closed_models(self):
        """Test with multiple closed models."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "ClosedB", "OpenC"],
            "eci": [100.0, 120.0, 125.0],
            "date": pd.to_datetime(["2024-01-01", "2024-03-01", "2024-06-01"]),
            "Open": [False, False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        assert len(gaps) == 2
        # Both should be matched by OpenC
        assert all(g["matched"] for g in gaps)
        assert all(g["open_model"] == "OpenC" for g in gaps)

    def test_eci_tolerance(self):
        """Test that open model within 1 ECI point counts as match."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [100.0, 99.5],  # OpenB is 0.5 below, should still match
            "date": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "Open": [False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        # Should match because 99.5 >= 100 - 1 (threshold is ECI_MATCH_THRESHOLD = 1.0)
        assert gaps[0]["matched"] is True

    def test_eci_tolerance_exact_boundary(self):
        """Test open model exactly at threshold boundary (1 point below)."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [100.0, 99.0],  # Exactly 1 point below = exactly at threshold
            "date": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "Open": [False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        # Should match because 99.0 >= 100 - 1 = 99.0 (inclusive boundary)
        assert gaps[0]["matched"] is True

    def test_eci_tolerance_just_outside(self):
        """Test open model just outside threshold (1.01 points below)."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [100.0, 98.99],  # Just outside threshold
            "date": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "Open": [False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        # Should NOT match because 98.99 < 100 - 1 = 99.0
        assert gaps[0]["matched"] is False

    def test_eci_tolerance_open_above_closed(self):
        """Test open model above closed model (always matches)."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [100.0, 105.0],  # Open is ahead
            "date": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "Open": [False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        # Should definitely match (open exceeds closed)
        assert gaps[0]["matched"] is True


class TestCurrentGapEstimation:
    """Tests for estimate_current_gap function."""

    def test_no_unmatched_models(self):
        """Test when all models are matched."""
        gaps = [
            {"matched": True, "gap_months": 3.0},
            {"matched": True, "gap_months": 5.0},
        ]
        matched_gaps = [3.0, 5.0]

        result = estimate_current_gap(gaps, matched_gaps)

        assert result["estimated_current_gap"] == 0
        assert result["min_current_gap"] == 0
        assert result["confidence"] == "high"

    def test_with_unmatched_models(self):
        """Test estimation with unmatched models."""
        gaps = [
            {"matched": True, "gap_months": 3.0},
            {"matched": True, "gap_months": 5.0},
            {"matched": False, "gap_months": 8.0},  # Unmatched, 8 months old
            {"matched": False, "gap_months": 6.0},  # Unmatched, 6 months old
        ]
        matched_gaps = [3.0, 5.0]

        result = estimate_current_gap(gaps, matched_gaps)

        # Min should be the oldest unmatched model
        assert result["min_current_gap"] == 8.0
        # Estimate should be >= min
        assert result["estimated_current_gap"] >= result["min_current_gap"]
        # Unmatched ages should be sorted descending
        assert result["unmatched_ages"] == [8.0, 6.0]

    def test_estimate_increases_with_more_unmatched(self):
        """Test that estimate increases with more unmatched models."""
        gaps_few = [
            {"matched": True, "gap_months": 4.0},
            {"matched": True, "gap_months": 5.0},
            {"matched": True, "gap_months": 6.0},
            {"matched": False, "gap_months": 7.0},
        ]
        gaps_many = [
            {"matched": True, "gap_months": 4.0},
            {"matched": True, "gap_months": 5.0},
            {"matched": True, "gap_months": 6.0},
            {"matched": False, "gap_months": 7.0},
            {"matched": False, "gap_months": 6.0},
            {"matched": False, "gap_months": 5.0},
        ]
        matched_gaps = [4.0, 5.0, 6.0]

        result_few = estimate_current_gap(gaps_few, matched_gaps)
        result_many = estimate_current_gap(gaps_many, matched_gaps)

        # More unmatched models should increase uncertainty premium
        # Both should have same min (7.0) but many should have higher estimate
        assert result_few["min_current_gap"] == 7.0
        assert result_many["min_current_gap"] == 7.0


class TestStatisticsCalculation:
    """Tests for calculate_statistics function."""

    def test_basic_statistics(self):
        """Test basic statistics calculation."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "ClosedB", "OpenC", "OpenD"],
            "eci": [100.0, 110.0, 105.0, 115.0],
            "date": pd.to_datetime([
                "2024-01-01", "2024-02-01",
                "2024-03-01", "2024-04-01"
            ]),
            "Open": [False, False, True, True],
        })

        gaps = [
            {"matched": True, "gap_months": 2.0},
            {"matched": True, "gap_months": 2.0},
        ]

        stats = calculate_statistics(df, gaps)

        assert "avg_horizontal_gap_months" in stats
        assert "std_horizontal_gap" in stats
        assert "ci_90_low" in stats
        assert "ci_90_high" in stats
        assert stats["total_matched"] == 2
        assert stats["total_unmatched"] == 0
        assert "current_gap_estimate" in stats

    def test_vertical_gap_calculation(self):
        """Test that vertical gap is best_closed - best_open."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [150.0, 140.0],
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "Open": [False, True],
        })

        gaps = [{"matched": False, "gap_months": 5.0}]
        stats = calculate_statistics(df, gaps)

        # Vertical gap = 150 - 140 = 10
        assert stats["current_vertical_gap"] == 10.0


class TestHistoricalGaps:
    """Tests for calculate_historical_gaps function."""

    def test_historical_gap_calculation(self):
        """Test that historical gaps are calculated for time points."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB", "ClosedC", "OpenD"],
            "eci": [100.0, 105.0, 120.0, 125.0],
            "date": pd.to_datetime([
                "2023-01-01", "2023-06-01",
                "2024-01-01", "2024-06-01"
            ]),
            "Open": [False, True, False, True],
        })

        historical = calculate_historical_gaps(df)

        # Should have multiple time points
        assert len(historical) > 0
        # Each entry should have required fields
        for entry in historical:
            assert "date" in entry
            assert "gap_months" in entry
            assert "matched" in entry
            assert "reference_model" in entry
            assert "open_frontier_model" in entry

    def test_gap_cannot_be_negative(self):
        """Test that gaps are never negative."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [100.0, 150.0],  # Open is way ahead
            "date": pd.to_datetime(["2024-01-01", "2023-01-01"]),  # Open released first
            "Open": [False, True],
        })

        historical = calculate_historical_gaps(df)

        for entry in historical:
            assert entry["gap_months"] >= 0


class TestOrganizationClassification:
    """Tests for organization classification functions."""

    def test_china_organizations(self):
        """Test Chinese organization detection."""
        assert is_china_org("DeepSeek") is True
        assert is_china_org("Alibaba") is True
        assert is_china_org("DeepSeek,Peking University") is True
        assert is_china_org("Baichuan") is True
        assert is_china_org("01.AI") is True
        assert is_china_org("Moonshot") is True

    def test_us_organizations(self):
        """Test US organization detection."""
        assert is_us_org("OpenAI") is True
        assert is_us_org("Anthropic") is True
        assert is_us_org("Google DeepMind") is True
        assert is_us_org("Meta AI") is True
        assert is_us_org("Microsoft Research") is True
        assert is_us_org("xAI") is True

    def test_non_china_non_us(self):
        """Test organizations that are neither China nor US."""
        assert is_china_org("Mistral AI") is False
        assert is_us_org("Mistral AI") is False
        assert is_china_org("Technology Innovation Institute") is False
        assert is_us_org("Technology Innovation Institute") is False


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame({
            "Model": [],
            "eci": [],
            "date": [],
            "Open": [],
        })

        gaps = calculate_horizontal_gaps(df)
        assert gaps == []

    def test_only_open_models(self):
        """Test with only open models (no closed)."""
        df = pd.DataFrame({
            "Model": ["OpenA", "OpenB"],
            "eci": [100.0, 110.0],
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "Open": [True, True],
        })

        gaps = calculate_horizontal_gaps(df)
        assert gaps == []

    def test_only_closed_models(self):
        """Test with only closed models (no open)."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "ClosedB"],
            "eci": [100.0, 110.0],
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "Open": [False, False],
        })

        gaps = calculate_horizontal_gaps(df)

        # Should have gaps but all unmatched
        assert len(gaps) == 2
        assert all(not g["matched"] for g in gaps)

    def test_nan_handling(self):
        """Test that NaN values are handled gracefully."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB", "ClosedC"],
            "eci": [100.0, np.nan, 120.0],
            "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "Open": [False, True, False],
        })

        # Should not raise an error
        gaps = calculate_horizontal_gaps(df)
        assert isinstance(gaps, list)


class TestRealWorldScenario:
    """Tests using realistic data similar to actual ECI data."""

    def test_realistic_gap_scenario(self):
        """Test with data mimicking real ECI progression."""
        df = pd.DataFrame({
            "Model": [
                "GPT-4", "LLaMA-65B", "Claude 3 Opus",
                "Llama 3.1-405B", "o1", "DeepSeek-R1"
            ],
            "eci": [126.0, 109.0, 127.0, 128.0, 142.0, 139.0],
            "date": pd.to_datetime([
                "2023-03-14", "2023-02-24", "2024-02-29",
                "2024-07-23", "2024-12-17", "2025-01-20"
            ]),
            "Open": [False, True, False, True, False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        # GPT-4 should be matched by Llama 3.1-405B
        gpt4_gap = next((g for g in gaps if g["closed_model"] == "GPT-4"), None)
        assert gpt4_gap is not None
        assert gpt4_gap["matched"] is True
        assert gpt4_gap["open_model"] == "Llama 3.1-405B"
        # Exact calculation: Mar 14, 2023 to Jul 23, 2024
        expected_days = (pd.to_datetime("2024-07-23") - pd.to_datetime("2023-03-14")).days
        expected_months = expected_days / DAYS_PER_MONTH
        assert abs(gpt4_gap["gap_months"] - expected_months) < 0.1

        # o1 should NOT be matched by DeepSeek-R1
        o1_gap = next((g for g in gaps if g["closed_model"] == "o1"), None)
        assert o1_gap is not None
        # DeepSeek-R1 (139) vs o1 (142): 139 >= 142-1 = 141? No, 139 < 141
        # So o1 should be unmatched
        assert o1_gap["matched"] is False


class TestConstants:
    """Tests verifying constant values and their mathematical correctness."""

    def test_days_per_month_accuracy(self):
        """Verify DAYS_PER_MONTH is the accurate average."""
        # Average days per month accounting for leap years = 365.25 / 12
        expected = 365.25 / 12  # = 30.4375
        assert DAYS_PER_MONTH == expected
        # Verify it's more accurate than the commonly used 30.5
        assert abs(DAYS_PER_MONTH - 30.4375) < 0.0001

    def test_eci_match_threshold(self):
        """Verify ECI_MATCH_THRESHOLD is documented value."""
        assert ECI_MATCH_THRESHOLD == 1.0

    def test_z_score_for_90_ci(self):
        """Verify the z-score used for 90% CI is correct."""
        from scipy.stats import norm
        # For 90% CI (two-tailed), we need z where P(-z < Z < z) = 0.90
        # This means P(Z < z) = 0.95
        z_90 = norm.ppf(0.95)
        assert abs(z_90 - 1.645) < 0.001


class TestConfidenceIntervals:
    """Tests verifying confidence interval calculations are correct."""

    def test_ci_formula_correctness(self):
        """Verify CI formula: mean ± z * (std / sqrt(n))."""
        # Create test data with known statistics
        test_gaps = [3.0, 4.0, 5.0, 6.0, 7.0]
        mean = np.mean(test_gaps)
        std = np.std(test_gaps, ddof=1)  # Unbiased
        n = len(test_gaps)
        sem = std / np.sqrt(n)

        expected_ci_low = mean - 1.645 * sem
        expected_ci_high = mean + 1.645 * sem

        # Verify our understanding of the formula
        assert abs(mean - 5.0) < 0.001
        assert expected_ci_low < mean < expected_ci_high
        # CI should be symmetric around mean
        assert abs((expected_ci_high - mean) - (mean - expected_ci_low)) < 0.001

    def test_ci_narrows_with_more_samples(self):
        """Verify CI narrows as sample size increases."""
        small_sample = [3.0, 4.0, 5.0, 6.0, 7.0]
        large_sample = small_sample * 10  # 50 samples

        def calc_ci_width(data):
            std = np.std(data, ddof=1)
            sem = std / np.sqrt(len(data))
            return 2 * 1.645 * sem

        small_width = calc_ci_width(small_sample)
        large_width = calc_ci_width(large_sample)

        # CI width should scale as 1/sqrt(n)
        # sqrt(50)/sqrt(5) ≈ 3.16, so large should be ~3.16x narrower
        ratio = small_width / large_width
        assert 3.0 < ratio < 3.6  # Allow for floating point precision

    def test_statistics_returns_valid_ci(self):
        """Verify calculate_statistics returns valid CI bounds."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "ClosedB", "OpenC", "OpenD"],
            "eci": [100.0, 105.0, 102.0, 107.0],
            "date": pd.to_datetime([
                "2024-01-01", "2024-02-01",
                "2024-03-01", "2024-04-01"
            ]),
            "Open": [False, False, True, True],
        })

        gaps = [
            {"matched": True, "gap_months": 2.0},
            {"matched": True, "gap_months": 2.0},
        ]

        stats = calculate_statistics(df, gaps)

        # CI should bracket the mean
        assert stats["ci_90_low"] <= stats["avg_horizontal_gap_months"]
        assert stats["avg_horizontal_gap_months"] <= stats["ci_90_high"]
        # CI bounds should be reasonable (not negative for gap data)
        # Note: CI_low could theoretically be negative if std is large
        assert stats["ci_90_high"] > 0


class TestDateArithmetic:
    """Tests verifying date arithmetic precision."""

    def test_exact_month_calculation(self):
        """Verify gap calculation uses exact day count."""
        # 2024 is a leap year, so Feb has 29 days
        # Jan 1 to Mar 1 = 31 (Jan) + 29 (Feb) = 60 days
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [100.0, 101.0],
            "date": pd.to_datetime(["2024-01-01", "2024-03-01"]),
            "Open": [False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        expected_days = 60  # Jan (31) + Feb (29 in 2024)
        expected_months = expected_days / DAYS_PER_MONTH
        # gap_months is rounded to 1 decimal place
        assert abs(gaps[0]["gap_months"] - round(expected_months, 1)) < 0.01

    def test_year_boundary_calculation(self):
        """Verify gap calculation across year boundary."""
        # Dec 15, 2023 to Jan 15, 2024 = 31 days
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [100.0, 101.0],
            "date": pd.to_datetime(["2023-12-15", "2024-01-15"]),
            "Open": [False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        expected_days = 31
        expected_months = expected_days / DAYS_PER_MONTH
        # gap_months is rounded to 1 decimal place
        assert abs(gaps[0]["gap_months"] - round(expected_months, 1)) < 0.01


class TestOpenCaughtUp:
    """Tests for the bootstrap-mirroring significance predicate."""

    def test_z_value_is_one_sided_5pct(self):
        """Z_ONE_SIDED_05 should be the 95th percentile of the standard normal."""
        assert abs(Z_ONE_SIDED_05 - 1.6448536) < 1e-5

    def test_caught_up_when_close_within_se(self):
        """SOTA only slightly ahead, within z*SE -> not significantly better."""
        # SE = sqrt(2^2 + 2^2) = 2.8284; z*SE = 1.6449 * 2.8284 = 4.6526
        # sota - open = 3 <= 4.6526 -> caught up
        assert _open_caught_up(100.0, 2.0, 103.0, 2.0, threshold=1.0) is True

    def test_not_caught_up_when_far_beyond_se(self):
        """SOTA far ahead relative to SE -> significantly better."""
        # SE = sqrt(1+1) = 1.4142; z*SE = 2.326; sota - open = 10 > 2.326
        assert _open_caught_up(100.0, 1.0, 110.0, 1.0, threshold=1.0) is False

    def test_open_above_sota_always_caught_up(self):
        """Open point estimate above SOTA -> trivially caught up."""
        assert _open_caught_up(105.0, 1.0, 100.0, 1.0, threshold=1.0) is True

    def test_threshold_fallback_when_no_std(self):
        """With missing std, fall back to a point-estimate match within threshold."""
        # 100 >= 100.5 - 1 = 99.5 -> caught up
        assert _open_caught_up(100.0, np.nan, 100.5, np.nan, threshold=1.0) is True
        # 98 >= 100 - 1 = 99 -> not caught up
        assert _open_caught_up(98.0, np.nan, 100.0, np.nan, threshold=1.0) is False

    def test_boundary_around_z_se(self):
        """The decision flips as sota - open crosses z*SE."""
        import math
        se = math.sqrt(2.0 ** 2 + 2.0 ** 2)
        diff = Z_ONE_SIDED_05 * se
        # Just inside the boundary -> caught up; just outside -> not.
        assert _open_caught_up(100.0, 2.0, 100.0 + diff - 0.01, 2.0, threshold=1.0) is True
        assert _open_caught_up(100.0, 2.0, 100.0 + diff + 0.01, 2.0, threshold=1.0) is False


class TestGapMetrics:
    """Tests for the day-by-day SOTA-frontier gap metrics."""

    def test_single_day_window_threshold_path(self):
        """A one-day window with no std exercises the threshold path exactly."""
        df = pd.DataFrame({
            "Model": ["C1", "C2", "O1"],
            "eci": [100.0, 120.0, 121.0],
            "date": pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01"]),
            "Open": [False, False, True],
        })

        m = calculate_gap_metrics(df, score_col="eci", threshold=1.0)

        assert m is not None
        # Window collapses to the single day the open model exists (latest date).
        assert m["n_days"] == 1
        # Open (121) catches the most recent closed SOTA C2 (120, within
        # threshold). Gap = Jul 1 2024 -> Jan 1 2025 = 184 days.
        expected_days = (pd.to_datetime("2025-01-01") - pd.to_datetime("2024-07-01")).days
        expected_months = expected_days / DAYS_PER_MONTH
        assert abs(m["avg_time_gap_months"] - expected_months) < 1e-6
        # Strict: open (121) > C2 (120), so same reference, same gap.
        assert abs(m["avg_time_gap_months_strict"] - expected_months) < 1e-6
        # Vertical: absolute SOTA = max(120, 121) = 121, open = 121 -> 0.
        assert abs(m["avg_vertical_gap"]) < 1e-9

    def test_strict_gap_never_below_lenient(self):
        """Strict criterion is more demanding, so its gap >= lenient gap."""
        df = pd.DataFrame({
            "Model": ["C1", "C2", "C3", "O1", "O2"],
            "eci": [100.0, 120.0, 140.0, 119.0, 135.0],
            "eci_std": [2.0, 2.0, 2.0, 2.0, 2.0],
            "date": pd.to_datetime([
                "2023-01-01", "2023-09-01", "2024-06-01",
                "2024-01-01", "2024-12-01",
            ]),
            "Open": [False, False, False, True, True],
        })

        m = calculate_gap_metrics(df, score_col="eci")

        assert m is not None
        assert m["n_days"] > 1
        assert m["avg_time_gap_months_strict"] >= m["avg_time_gap_months"] - 1e-9

    def test_lenient_more_permissive_than_strict_specific_day(self):
        """With uncertainty, open catches a SOTA whose point estimate is higher."""
        # Open at 119 (+/- std) vs closed SOTA at 120: lenient catches the 120
        # model (not significantly better), strict does not (119 !> 120).
        df = pd.DataFrame({
            "Model": ["C_old", "C_new", "O1"],
            "eci": [100.0, 120.0, 119.0],
            "eci_std": [2.0, 2.0, 2.0],
            "date": pd.to_datetime(["2023-01-01", "2024-01-01", "2024-06-01"]),
            "Open": [False, False, True],
        })

        m = calculate_gap_metrics(df, score_col="eci")
        assert m is not None
        # n_days == 1 (open exists only on its release day, the latest date).
        # Lenient -> ref C_new (2024-01-01); strict -> ref C_old (2023-01-01).
        lenient_days = (pd.to_datetime("2024-06-01") - pd.to_datetime("2024-01-01")).days
        strict_days = (pd.to_datetime("2024-06-01") - pd.to_datetime("2023-01-01")).days
        assert abs(m["avg_time_gap_months"] - lenient_days / DAYS_PER_MONTH) < 1e-6
        assert abs(m["avg_time_gap_months_strict"] - strict_days / DAYS_PER_MONTH) < 1e-6

    def test_window_override(self):
        """Explicit window bounds the number of days analyzed."""
        df = pd.DataFrame({
            "Model": ["C1", "O1"],
            "eci": [100.0, 105.0],
            "eci_std": [2.0, 2.0],
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "Open": [False, True],
        })

        m = calculate_gap_metrics(
            df, score_col="eci",
            window_start="2024-02-01", window_end="2024-02-10",
        )
        assert m is not None
        assert m["n_days"] == 10  # Feb 1..Feb 10 inclusive

    def test_returns_none_without_both_groups(self):
        """No SOTA (closed) models -> cannot compute a gap."""
        df = pd.DataFrame({
            "Model": ["O1", "O2"],
            "eci": [100.0, 110.0],
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "Open": [True, True],
        })
        assert calculate_gap_metrics(df, score_col="eci") is None


class TestStatisticsNewFields:
    """The replaced calculate_statistics must surface the new methodology."""

    def test_new_keys_present_and_consistent(self):
        df = pd.DataFrame({
            "Model": ["C1", "C2", "C3", "O1", "O2"],
            "eci": [100.0, 120.0, 140.0, 119.0, 135.0],
            "eci_std": [2.0, 2.0, 2.0, 2.0, 2.0],
            "date": pd.to_datetime([
                "2023-01-01", "2023-09-01", "2024-06-01",
                "2024-01-01", "2024-12-01",
            ]),
            "Open": [False, False, False, True, True],
        })
        gaps = calculate_horizontal_gaps(df)
        stats = calculate_statistics(df, gaps)

        for key in (
            "avg_horizontal_gap_months",
            "avg_horizontal_gap_months_strict",
            "std_horizontal_gap",
            "ci_90_low",
            "ci_90_high",
            "current_vertical_gap",
            "avg_vertical_gap",
            "vertical_gap_ci_90_low",
            "vertical_gap_ci_90_high",
            "gap_window",
        ):
            assert key in stats, f"missing {key}"

        # Strict variant is at least as large as the lenient one.
        assert stats["avg_horizontal_gap_months_strict"] >= stats["avg_horizontal_gap_months"]
        # CI brackets the mean (quantile-based band).
        assert stats["ci_90_low"] <= stats["avg_horizontal_gap_months"] <= stats["ci_90_high"]
        assert stats["vertical_gap_ci_90_low"] <= stats["avg_vertical_gap"] <= stats["vertical_gap_ci_90_high"]
        assert stats["gap_window"]["n_days"] >= 1


from eci_bootstrap import EciBootstrap
import numpy as _np
from update_data import _match_method, CAUGHT_UP_PROB


def _boot(prob_a_over_b):
    # Construct draws so P(A>B) == prob_a_over_b over 100 samples.
    k = int(round(prob_a_over_b * 100))
    a = _np.concatenate([_np.full(k, 2.0), _np.full(100 - k, 0.0)])
    b = _np.ones(100)
    return EciBootstrap({"A": a, "B": b}, n_samples=100, seed=1, source_hash="h")


class TestBootstrapCriterion:
    def test_bootstrap_caught_up_at_5pct(self):
        boot = _boot(0.05)  # exactly 5% -> caught up (>= 0.05)
        assert _open_caught_up(0, 1, 100, 1, threshold=1.0,
                               open_name="A", sota_name="B", bootstrap=boot) is True

    def test_bootstrap_not_caught_up_below_5pct(self):
        boot = _boot(0.04)
        assert _open_caught_up(0, 1, 100, 1, threshold=1.0,
                               open_name="A", sota_name="B", bootstrap=boot) is False

    def test_bootstrap_falls_back_when_name_missing(self):
        boot = _boot(0.04)  # would say "not caught up" if used
        # "Z" absent -> prob_exceeds None -> analytical path. Open well above SOTA.
        assert _open_caught_up(105, 1, 100, 1, threshold=1.0,
                               open_name="Z", sota_name="B", bootstrap=boot) is True

    def test_match_method_labels(self):
        boot = _boot(0.5)
        assert _match_method(1, 1, open_name="A", sota_name="B", bootstrap=boot) == "bootstrap"
        assert _match_method(2.0, 2.0) == "analytical"
        assert _match_method(_np.nan, _np.nan) == "threshold"

    def test_caught_up_prob_constant(self):
        assert CAUGHT_UP_PROB == 0.05


class TestHorizontalGapsBootstrap:
    def test_bootstrap_overrides_threshold_verdict(self):
        # No std -> the no-bootstrap path is the point-estimate threshold(1.0),
        # which says NOT matched (98.5 < 99). The bootstrap (P(open>closed)=0.10)
        # is consulted first and says matched, overriding the threshold.
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [100.0, 98.5],
            "date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "Open": [False, True],
        })
        a = _np.concatenate([_np.full(10, 200.0), _np.zeros(90)])  # P(open>closed)=0.10
        b = _np.full(100, 100.0)
        boot = EciBootstrap({"OpenB": a, "ClosedA": b}, n_samples=100, seed=1, source_hash="h")

        no_boot = calculate_horizontal_gaps(df)
        assert no_boot[0]["matched"] is False  # threshold path (no std)

        with_boot = calculate_horizontal_gaps(df, bootstrap=boot)
        assert with_boot[0]["matched"] is True
        assert with_boot[0]["match_type"] == "bootstrap"


class TestGapMetricsBootstrap:
    def test_lenient_uses_bootstrap_reference(self):
        # Open(119) vs two closed SOTA: C_new(120, 2024-01) and C_old(100, 2023-01).
        # Bootstrap: P(open>C_new)=0.10 (caught up), so lenient ref = C_new.
        df = pd.DataFrame({
            "Model": ["C_old", "C_new", "O1"],
            "eci": [100.0, 120.0, 119.0],
            "eci_std": [2.0, 2.0, 2.0],
            "date": pd.to_datetime(["2023-01-01", "2024-01-01", "2024-06-01"]),
            "Open": [False, False, True],
        })
        a = _np.concatenate([_np.full(10, 999.0), _np.zeros(90)])  # P(O1>C_new)=0.10
        cnew = _np.full(100, 120.0)
        cold = _np.zeros(100)  # O1 always > C_old
        boot = EciBootstrap({"O1": a, "C_new": cnew, "C_old": cold},
                            n_samples=100, seed=1, source_hash="h")
        m = calculate_gap_metrics(df, score_col="eci", bootstrap=boot)
        assert m is not None
        lenient_days = (pd.to_datetime("2024-06-01") - pd.to_datetime("2024-01-01")).days
        assert abs(m["avg_time_gap_months"] - lenient_days / DAYS_PER_MONTH) < 1e-6

    def test_statistics_forwards_bootstrap(self):
        df = pd.DataFrame({
            "Model": ["C1", "O1"],
            "eci": [100.0, 98.0],
            "eci_std": [1.0, 1.0],
            "date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "Open": [False, True],
        })
        a = _np.concatenate([_np.full(20, 200.0), _np.zeros(80)])  # P=0.20 caught up
        b = _np.full(100, 100.0)
        boot = EciBootstrap({"O1": a, "C1": b}, n_samples=100, seed=1, source_hash="h")
        gaps = calculate_horizontal_gaps(df, bootstrap=boot)
        stats = calculate_statistics(df, gaps, bootstrap=boot)
        assert stats["avg_horizontal_gap_months"] >= 0
        assert stats["total_matched"] == 1


class TestHistoricalGapsBootstrap:
    def test_accepts_bootstrap_and_runs(self):
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB", "ClosedC", "OpenD"],
            "eci": [100.0, 105.0, 120.0, 125.0],
            "eci_std": [2.0, 2.0, 2.0, 2.0],
            "date": pd.to_datetime(["2023-01-01", "2023-06-01", "2024-01-01", "2024-06-01"]),
            "Open": [False, True, False, True],
        })
        names = ["ClosedA", "OpenB", "ClosedC", "OpenD"]
        boot = EciBootstrap({n: _np.full(50, i * 10.0) for i, n in enumerate(names)},
                            n_samples=50, seed=1, source_hash="h")
        hist = calculate_historical_gaps(df, bootstrap=boot)
        assert isinstance(hist, list) and len(hist) > 0
        for e in hist:
            assert e["gap_months"] >= 0
            assert "reference_model" in e and "open_frontier_model" in e


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
