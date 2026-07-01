# Threshold-Crossing Gap Analysis Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port open_closed_gap's threshold-crossing gap methodology into this repo's live pipeline as a new `scripts/threshold_gap.py` module, emitting additive `threshold_analysis` / `threshold_aggregate` sections in `data.json`.

**Architecture:** Pure calculation module (`scripts/threshold_gap.py`, numpy/pandas only, no I/O) consumed by `scripts/update_data.py` at the three benchmark-processing sites plus one aggregate site. This repo's classification, live data, ×100 score scale, and months units prevail; the source repo's threshold-grid/first-crosser/pair-dedup/validity-review/Gaussian-smoother logic is ported. Spec: `docs/specs/2026-07-01-threshold-gap-integration-design.md`.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest.

---

### Task 1: Core datapoint computation (`compute_threshold_datapoints`)

**Files:**
- Create: `scripts/threshold_gap.py`
- Create: `tests/test_threshold_gap.py`

- [ ] **Step 1: Write failing tests for grid/first-crosser/gap/still-open**

Create `tests/test_threshold_gap.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_threshold_gap.py -v`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'threshold_gap'`

- [ ] **Step 3: Write the module core**

Create `scripts/threshold_gap.py`:

```python
"""Threshold-crossing gap analysis, ported from Havard Ihle's open_closed_gap
(https://github.com/htihle/open_closed_gap).

For each score threshold T on a fixed grid: gap = release date of the first
open-weight model to score >= T minus release date of the first closed model
to do so. Where the source repo's logic conflicts with this repo's (open/
closed classification, live data, percentage score scale, month units), this
repo's logic prevails. See
docs/specs/2026-07-01-threshold-gap-integration-design.md.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 365.25 / 12  # matches scripts/update_data.py

# Gaussian-smoother parameters — the source repo's published-figure settings.
SMOOTH_BANDWIDTH_DAYS = 60.0
SMOOTH_STEP_DAYS = 7
SMOOTH_N_BOOT = 5000
SMOOTH_CI = 0.90
SMOOTH_MIN_ESS = 2.0
SMOOTH_SEED = 0

# Per-benchmark verdicts from the source repo's manual per-pair review,
# mapped to this repo's benchmark ids. Score thresholds are on THIS repo's
# scale (0-1 benchmarks stored as percentages, so source values x100; METR
# horizons are minutes in both). keep=False benchmarks still get per-benchmark
# datapoints but are excluded from the cross-benchmark aggregate (source
# DISCARD verdicts; ECI excluded as a composite index).
THRESHOLD_REVIEW: dict[str, dict] = {
    "gpqa_diamond": {
        "accepted_thresholds": [35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0,
                                70.0, 75.0, 80.0, 85.0, 90.0],
        "validity_floor": 20.0,
        "data_access": "public",
        "keep": True,
        "source_review": "open_closed_gap",
    },
    "math_level_5": {
        "accepted_thresholds": [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0,
                                50.0, 55.0, 60.0, 65.0, 80.0, 90.0, 95.0],
        "validity_floor": 10.0,
        "data_access": "public",
        "keep": True,
        "source_review": "open_closed_gap",
    },
    "otis_mock_aime": {
        "accepted_thresholds": [5.0, 10.0, 15.0, 20.0, 45.0, 50.0, 55.0,
                                60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0,
                                95.0],
        "validity_floor": 5.0,
        "data_access": "public",
        "keep": True,
        "source_review": "open_closed_gap",
    },
    "frontiermath_public": {
        "accepted_thresholds": [10.0, 15.0, 25.0, 30.0, 35.0],
        "validity_floor": 10.0,
        "data_access": "private",
        "keep": True,
        "source_review": "open_closed_gap",
    },
    "metr_time_horizon": {
        "thresholds": [2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
        "accepted_thresholds": [2.0, 5.0, 10.0, 20.0, 50.0],
        "validity_floor": 1.5,
        "data_access": "private",
        "keep": True,
        "source_review": "open_closed_gap",
    },
    # Source verdict: DISCARD (excluded from its aggregate).
    "swe_bench_verified": {
        "data_access": "public",
        "keep": False,
        "source_review": "open_closed_gap",
    },
    # Source verdict: reviewed, 0 datapoints accepted.
    "simpleqa_verified": {
        "validity_floor": 20.0,
        "data_access": "public",
        "keep": False,
        "source_review": "open_closed_gap",
    },
    # Source verdict: DISCARD.
    "chess_puzzles": {
        "data_access": "private",
        "keep": False,
        "source_review": "open_closed_gap",
    },
    # Not a source benchmark: composite index, would double-count its
    # constituent benchmarks in the aggregate.
    "eci": {
        "data_access": None,
        "keep": False,
        "source_review": None,
    },
}


def auto_thresholds(scores: pd.Series) -> np.ndarray:
    s_max = float(scores.max())
    if s_max <= 1.05:
        return np.arange(0.05, 1.0, 0.05)
    if s_max <= 110:
        return np.arange(5.0, 100.0, 5.0)
    s_min = float(max(scores.min(), 0))
    return np.linspace(s_min, s_max, 21)[1:-1]


def auto_validity_floor(scores: pd.Series) -> float:
    """Below the 10th percentile of observed scores, "first crosser" mostly
    reflects test order rather than capability (source heuristic)."""
    return float(np.percentile(scores, 10))


def first_crosser(df: pd.DataFrame, threshold: float,
                  score_col: str = "score", date_col: str = "date"):
    """Earliest-dated row with score >= threshold, or None."""
    hits = df[df[score_col] >= threshold]
    if hits.empty:
        return None
    return hits.sort_values(date_col, kind="mergesort").iloc[0]


def compute_threshold_datapoints(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    model_col: str = "model",
    date_col: str = "date",
    open_col: str = "Open",
    thresholds=None,
    accepted_thresholds=None,
    validity_floor: float | None = None,
    as_of=None,
) -> list[dict]:
    """One dict per grid threshold a closed model has reached.

    Expects `open_col` to hold this repo's open/closed classification.
    Dates in the returned dicts are pandas Timestamps (callers serialise).
    """
    d = df.dropna(subset=[date_col, score_col]).copy()
    if d.empty or open_col not in d.columns:
        return []
    is_open = d[open_col].astype(bool)
    closed = d[~is_open]
    open_df = d[is_open]
    if closed.empty or open_df.empty:
        return []

    if thresholds is None:
        thresholds = auto_thresholds(d[score_col])
    thresholds = np.asarray(list(thresholds), dtype=float)
    if thresholds.size == 0:
        return []

    floor = validity_floor
    if floor is None:
        floor = auto_validity_floor(d[score_col])

    allow = None
    if accepted_thresholds is not None:
        allow = {round(float(x), 6) for x in accepted_thresholds}

    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(datetime.now())

    rows: list[dict] = []
    for t in thresholds:
        c = first_crosser(closed, float(t), score_col, date_col)
        if c is None:
            continue  # not reached by closed side either
        o = first_crosser(open_df, float(t), score_col, date_col)
        if o is None:
            # Closed reached T; open pending — ongoing gap vs as_of.
            gap_so_far_days = (as_of_ts - c[date_col]).days
            rows.append({
                "threshold": float(t),
                "first_closed_model": str(c[model_col]),
                "first_closed_date": c[date_col],
                "first_open_model": None,
                "first_open_date": None,
                "gap_days": None,
                "gap_months": None,
                "gap_months_so_far": round(gap_so_far_days / DAYS_PER_MONTH, 1),
                "valid": False,
                "accepted": False,
                "still_open": True,
                "validity_floor": float(floor),
            })
            continue
        gap_days = int((o[date_col] - c[date_col]).days)
        if allow is not None:
            is_valid = round(float(t), 6) in allow
        else:
            is_valid = float(t) > floor
        rows.append({
            "threshold": float(t),
            "first_closed_model": str(c[model_col]),
            "first_closed_date": c[date_col],
            "first_open_model": str(o[model_col]),
            "first_open_date": o[date_col],
            "gap_days": gap_days,
            "gap_months": round(gap_days / DAYS_PER_MONTH, 1),
            "gap_months_so_far": None,
            "valid": bool(is_valid),
            "accepted": False,
            "still_open": False,
            "validity_floor": float(floor),
        })

    # The same (closed, open) pair recurs across a contiguous run of
    # thresholds with an identical gap. Keep ONE representative datapoint per
    # accepted pair, at the HIGHEST threshold the pair spans (even when that
    # exact threshold is not in the allowlist — the source expands accepted
    # anchor pairs to the pair's full span).
    rows.sort(key=lambda r: r["threshold"])
    accepted_keys = {
        (r["first_closed_model"], r["first_open_model"])
        for r in rows
        if r["valid"]
    }
    for key in accepted_keys:
        members = [
            r for r in rows
            if (r["first_closed_model"], r["first_open_model"]) == key
        ]
        members[-1]["accepted"] = True
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_threshold_gap.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/threshold_gap.py tests/test_threshold_gap.py
git commit -m "feat: port threshold-crossing gap datapoint computation from open_closed_gap"
```

---

### Task 2: Validity allowlist + pair dedup semantics

**Files:**
- Modify: `tests/test_threshold_gap.py` (append tests; implementation already in Task 1)

- [ ] **Step 1: Write failing/passing verification tests for dedup + validity**

Append to `tests/test_threshold_gap.py`:

```python
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
```

- [ ] **Step 2: Run tests**

Run: `python3 -m pytest tests/test_threshold_gap.py -v`
Expected: all PASS (Task 1 implementation already covers these; any failure means the dedup port is wrong — fix `compute_threshold_datapoints`, not the tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_threshold_gap.py
git commit -m "test: cover validity allowlist and pair-dedup semantics of threshold gaps"
```

---

### Task 3: Gaussian-kernel smoother with bootstrap CI

**Files:**
- Modify: `scripts/threshold_gap.py` (append functions)
- Modify: `tests/test_threshold_gap.py` (append tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_threshold_gap.py`:

```python
from threshold_gap import gaussian_smooth_with_ci


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
        grid, mean, lo, hi = gaussian_smooth_with_ci(pd.Series(dtype="datetime64[ns]"),
                                                     pd.Series(dtype=float))
        assert len(grid) == 0 and len(mean) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_threshold_gap.py::TestGaussianSmoother -v`
Expected: FAIL — `ImportError: cannot import name 'gaussian_smooth_with_ci'`

- [ ] **Step 3: Implement (verbatim port of the source smoother)**

Append to `scripts/threshold_gap.py`:

```python
def _gaussian_smooth_grid(dates_int: np.ndarray, values: np.ndarray,
                          grid_int: np.ndarray, bandwidth: float,
                          min_ess: float = SMOOTH_MIN_ESS) -> np.ndarray:
    """Gaussian-kernel weighted mean at each grid point (vectorised).
    NaN where effective sample size sum(w) < min_ess."""
    dt = grid_int[:, None] - dates_int[None, :]  # (g, n)
    w = np.exp(-0.5 * (dt / bandwidth) ** 2)
    sw = w.sum(axis=1)
    num = (w * values[None, :]).sum(axis=1)
    out = np.full(sw.shape, np.nan, dtype=float)
    ok = sw >= min_ess
    out[ok] = num[ok] / sw[ok]
    return out


def gaussian_smooth_with_ci(
    dates: pd.Series,
    values: pd.Series,
    *,
    bandwidth_days: float = SMOOTH_BANDWIDTH_DAYS,
    step_days: int = SMOOTH_STEP_DAYS,
    n_boot: int = SMOOTH_N_BOOT,
    ci: float = SMOOTH_CI,
    min_ess: float = SMOOTH_MIN_ESS,
    seed: int = SMOOTH_SEED,
):
    """Nadaraya-Watson Gaussian-kernel smoother with bootstrap CI on a
    uniform date grid. Returns (grid_dates, mean, lo, hi) as pd.Series."""
    dates = pd.to_datetime(pd.Series(list(dates))).reset_index(drop=True)
    if dates.empty:
        empty_d = pd.Series([], dtype="datetime64[ns]")
        empty_f = pd.Series([], dtype=float)
        return empty_d, empty_f, empty_f.copy(), empty_f.copy()
    values = pd.Series(list(values)).reset_index(drop=True).astype(float).values

    d_int = dates.values.astype("datetime64[D]").astype(np.int64)
    grid = pd.date_range(dates.min(), dates.max(), freq=f"{step_days}D")
    grid_int = grid.values.astype("datetime64[D]").astype(np.int64)
    h = float(bandwidth_days)

    central = _gaussian_smooth_grid(d_int, values, grid_int, h, min_ess)

    n = len(values)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(grid_int)), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = _gaussian_smooth_grid(d_int[idx], values[idx], grid_int, h, min_ess)

    alpha = (1 - ci) / 2
    import warnings
    with warnings.catch_warnings():
        # All-NaN grid columns (outside every bootstrap's support) are expected.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        lo = np.nanquantile(boot, alpha, axis=0)
        hi = np.nanquantile(boot, 1 - alpha, axis=0)
    return pd.Series(grid), pd.Series(central), pd.Series(lo), pd.Series(hi)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_threshold_gap.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/threshold_gap.py tests/test_threshold_gap.py
git commit -m "feat: port Gaussian-kernel gap trend smoother with bootstrap CI"
```

---

### Task 4: JSON-ready builders (`build_threshold_analysis`, `build_threshold_aggregate`)

**Files:**
- Modify: `scripts/threshold_gap.py` (append)
- Modify: `tests/test_threshold_gap.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_threshold_gap.py`:

```python
from threshold_gap import (
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_threshold_gap.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_threshold_analysis'`

- [ ] **Step 3: Implement builders**

Append to `scripts/threshold_gap.py`:

```python
def _iso(ts) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat()


def summarize_datapoints(datapoints: list[dict]) -> dict:
    accepted = [r for r in datapoints if r["accepted"]]
    med = (round(float(np.median([r["gap_months"] for r in accepted])), 1)
           if accepted else None)
    return {
        "n_thresholds": len(datapoints),
        "n_valid": sum(1 for r in datapoints if r["valid"]),
        "n_accepted": len(accepted),
        "n_still_open": sum(1 for r in datapoints if r["still_open"]),
        "median_gap_months": med,
    }


def build_threshold_analysis(
    df: pd.DataFrame,
    benchmark_id: str,
    *,
    score_col: str = "score",
    model_col: str = "model",
    date_col: str = "date",
    open_col: str = "Open",
    as_of=None,
) -> dict:
    """JSON-ready `threshold_analysis` block for one benchmark."""
    review = THRESHOLD_REVIEW.get(benchmark_id, {})
    datapoints = compute_threshold_datapoints(
        df,
        score_col=score_col,
        model_col=model_col,
        date_col=date_col,
        open_col=open_col,
        thresholds=review.get("thresholds"),
        accepted_thresholds=review.get("accepted_thresholds"),
        validity_floor=review.get("validity_floor"),
        as_of=as_of,
    )
    out_points = [
        {
            **r,
            "first_closed_date": _iso(r["first_closed_date"]),
            "first_open_date": _iso(r["first_open_date"]),
        }
        for r in datapoints
    ]
    floor = (out_points[0]["validity_floor"] if out_points
             else review.get("validity_floor"))
    accepted = review.get("accepted_thresholds")
    config = {
        "keep": bool(review.get("keep", False)),
        "data_access": review.get("data_access"),
        "validity_floor": floor,
        "thresholds": ([float(x) for x in review["thresholds"]]
                       if review.get("thresholds") else None),
        "accepted_thresholds": ([float(x) for x in accepted]
                                if accepted else None),
        "source_review": review.get("source_review"),
    }
    return {
        "config": config,
        "datapoints": out_points,
        "summary": summarize_datapoints(datapoints),
    }


def build_threshold_aggregate(benchmarks: dict) -> dict:
    """Pool accepted datapoints from keep=True benchmarks; smooth the gap
    over time (overall + public/private split), mirroring the source's
    combined-by-access analysis. `benchmarks` maps benchmark id -> processed
    benchmark dict (with `metadata` and `threshold_analysis`)."""
    datapoints: list[dict] = []
    for bid, bench in benchmarks.items():
        ta = (bench or {}).get("threshold_analysis")
        if not ta or not ta.get("config", {}).get("keep"):
            continue
        access = ta["config"].get("data_access")
        name = (bench.get("metadata") or {}).get("name", bid)
        for r in ta["datapoints"]:
            if not r.get("accepted"):
                continue
            datapoints.append({
                "benchmark_id": bid,
                "benchmark_name": name,
                "data_access": access,
                "threshold": r["threshold"],
                "first_closed_model": r["first_closed_model"],
                "first_closed_date": r["first_closed_date"],
                "first_open_model": r["first_open_model"],
                "first_open_date": r["first_open_date"],
                "gap_days": r["gap_days"],
                "gap_months": r["gap_months"],
            })
    datapoints.sort(key=lambda p: (p["first_open_date"], p["benchmark_id"]))

    def trend_for(points: list[dict]):
        if len(points) < 2:
            return None
        grid, mean, lo, hi = gaussian_smooth_with_ci(
            pd.Series([p["first_open_date"] for p in points]),
            pd.Series([p["gap_months"] for p in points]),
        )
        out = []
        for g, m, l, h in zip(grid, mean, lo, hi):
            if not np.isfinite(m):
                continue
            out.append({
                "date": pd.Timestamp(g).date().isoformat(),
                "mean": round(float(m), 2),
                "lo": round(float(l), 2) if np.isfinite(l) else None,
                "hi": round(float(h), 2) if np.isfinite(h) else None,
            })
        if not out:
            return None
        return {
            "points": out,
            "n_datapoints": len(points),
            "n_benchmarks": len({p["benchmark_id"] for p in points}),
        }

    groups = {
        "overall": datapoints,
        "public": [p for p in datapoints if p["data_access"] == "public"],
        "private": [p for p in datapoints if p["data_access"] == "private"],
    }
    trends: dict = {}
    medians: dict = {}
    for key, pts in groups.items():
        t = trend_for(pts)
        if t is not None:
            trends[key] = t
        if pts:
            medians[key] = round(
                float(np.median([p["gap_months"] for p in pts])), 1)

    return {
        "datapoints": datapoints,
        "trends": trends,
        "medians": medians,
        "parameters": {
            "bandwidth_days": SMOOTH_BANDWIDTH_DAYS,
            "step_days": SMOOTH_STEP_DAYS,
            "n_boot": SMOOTH_N_BOOT,
            "ci": SMOOTH_CI,
            "min_ess": SMOOTH_MIN_ESS,
            "seed": SMOOTH_SEED,
        },
        "source": "open_closed_gap threshold-crossing methodology",
    }
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_threshold_gap.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/threshold_gap.py tests/test_threshold_gap.py
git commit -m "feat: add JSON-ready threshold-analysis and aggregate builders"
```

---

### Task 5: Wire into `scripts/update_data.py`

**Files:**
- Modify: `scripts/update_data.py` (import block ~line 95; `process_data` ~line 1204/1303; `process_benchmark_data` ~line 1415; `process_metr_data` ~line 1560; `process_all_benchmarks` ~line 1674)
- Modify: `tests/test_threshold_gap.py` (append integration test)

- [ ] **Step 1: Write failing test for the pipeline helper**

Append to `tests/test_threshold_gap.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_threshold_gap.py::TestUpdateDataIntegration -v`
Expected: FAIL — `ImportError: cannot import name 'compute_threshold_block'`

- [ ] **Step 3: Add import + helper to `update_data.py`**

After the ECI-bootstrap import block (after the `CAUGHT_UP_PROB` constant, ~line 99), insert:

```python
# Threshold-crossing gap analysis (ported from open_closed_gap; see
# docs/specs/2026-07-01-threshold-gap-integration-design.md).
try:
    from threshold_gap import build_threshold_analysis, build_threshold_aggregate
    THRESHOLD_GAP_AVAILABLE = True
except Exception as e:  # pragma: no cover - defensive
    build_threshold_analysis = None
    build_threshold_aggregate = None
    THRESHOLD_GAP_AVAILABLE = False
    print(f"Note: threshold gap module unavailable ({e})")


def compute_threshold_block(df, benchmark_id: str, *, score_col: str,
                            model_col: str) -> Optional[dict]:
    """Fail-open wrapper: threshold analysis must never kill the pipeline."""
    if not THRESHOLD_GAP_AVAILABLE:
        return None
    try:
        return build_threshold_analysis(
            df, benchmark_id, score_col=score_col, model_col=model_col)
    except Exception as e:
        logger.warning(f"Threshold analysis failed for {benchmark_id}: {e}")
        return None
```

Note: `compute_threshold_block` sits at module level (below the try/except), so
it must be defined AFTER `logger` (already at ~line 20) — placement after the
import block satisfies this.

- [ ] **Step 4: Attach per-benchmark blocks**

In `process_data()` — `df_combined` is built just before the frontier filter
(`df_frontier = df_combined[...]`). Add to the returned dict (alongside
`"historical_gaps": historical_gaps,`):

```python
        "threshold_analysis": compute_threshold_block(
            df_combined, "eci", score_col="eci", model_col="Model"),
```

In `process_all_benchmarks()`, inside the `benchmarks["eci"] = {...}` dict, add:

```python
        "threshold_analysis": eci_data["threshold_analysis"],
```

In `process_benchmark_data()` return dict, add:

```python
        "threshold_analysis": compute_threshold_block(
            df_combined, benchmark_id, score_col="score", model_col="model"),
```

In `process_metr_data()` return dict, add:

```python
        "threshold_analysis": compute_threshold_block(
            df_combined, "metr_time_horizon", score_col="score",
            model_col="model"),
```

- [ ] **Step 5: Attach the aggregate**

At the end of `process_all_benchmarks()`, replace the final `return {...}` with:

```python
    threshold_aggregate = None
    if THRESHOLD_GAP_AVAILABLE:
        try:
            threshold_aggregate = build_threshold_aggregate(benchmarks)
        except Exception as e:
            logger.warning(f"Threshold aggregate failed: {e}")

    return {
        "benchmarks": benchmarks,
        "default_benchmark": "eci",
        "threshold_aggregate": threshold_aggregate,
        "last_updated": datetime.now().isoformat(),
    }
```

- [ ] **Step 6: Run the full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: all PASS (pre-existing tests unaffected)

- [ ] **Step 7: Commit**

```bash
git add scripts/update_data.py tests/test_threshold_gap.py
git commit -m "feat: emit threshold_analysis and threshold_aggregate in data.json pipeline"
```

---

### Task 6: End-to-end verification against live data

**Files:** none created (verification only; `data.json` intentionally NOT committed — the daily workflow owns it)

- [ ] **Step 1: Run the pipeline end-to-end**

Run (from repo root): `cd /Users/theobearman/Documents/GitHub/ECI-Accessibility-Gap && python3 scripts/update_data.py`
Expected: exits 0; log lines show benchmarks processed; no `Threshold analysis failed` warnings.

- [ ] **Step 2: Inspect the new sections**

```bash
python3 - <<'EOF'
import json
d = json.load(open("data.json"))
agg = d["threshold_aggregate"]
print("aggregate datapoints:", len(agg["datapoints"]))
print("benchmarks pooled:", sorted({p["benchmark_id"] for p in agg["datapoints"]}))
print("medians:", agg["medians"])
print("trend keys:", list(agg["trends"].keys()))
for bid, b in d["benchmarks"].items():
    ta = b.get("threshold_analysis")
    print(bid, "->", (ta["summary"] if ta else None))
EOF
```

Expected: pooled benchmarks are a subset of {gpqa_diamond, math_level_5, otis_mock_aime, frontiermath_public, metr_time_horizon}; per-benchmark summaries look sane (accepted counts > 0 for KEEP benchmarks, medians positive single-to-double-digit months); `eci` has a summary but is absent from the pooled list.

- [ ] **Step 3: Sanity-check one benchmark by hand**

Pick one accepted GPQA datapoint and verify from the raw CSV that the named
models are genuinely the earliest crossers of that threshold (spot check).

- [ ] **Step 4: Restore `data.json`**

`data.json` is regenerated daily by CI; do not commit the locally regenerated
version. Run: `git checkout -- data.json` (only if it changed and the diff is
just the regeneration).

---

### Task 7: Gemini senior review (project CLAUDE.md protocol)

- [ ] **Step 1: Request review**

```bash
gemini -p "@scripts/threshold_gap.py You are a Senior Software Engineer with experience in research infrastructure. Review this code for: data handling errors, off-by-one bugs, incorrect aggregations, missing edge cases, and potential data leakage between conditions."
```

- [ ] **Step 2: Implement valid suggestions, re-run tests, commit fixes**

Evaluate each finding on its merits (verify before implementing). Re-run
`python3 -m pytest tests/ -v` after changes.

---

### Task 8: Documentation touch-up

**Files:**
- Modify: `README.md` (Key Features list)
- Create: `tasks/todo.md` review section (per user workflow)

- [ ] **Step 1: Add README bullet**

In README.md under **Key Features**, add:

```markdown
-   **Threshold-Crossing Analysis:** For each benchmark score threshold, how much later the first open model reached it than the first closed model (methodology ported from [open_closed_gap](https://github.com/htihle/open_closed_gap); published in `data.json` under `threshold_analysis` / `threshold_aggregate`).
```

- [ ] **Step 2: Commit**

```bash
git add README.md tasks/todo.md
git commit -m "docs: document threshold-crossing analysis integration"
```

---

## Self-Review Notes

- Spec coverage: datapoint core (T1), validity/dedup (T2), smoother (T3), JSON blocks + aggregate + keep/discard mapping (T4), pipeline wiring incl. fail-open (T5), e2e verification (T6), project review protocol (T7), docs (T8). Still-open semantics tested in T1; ×100-scaled review table in T1 module code.
- Types consistent: `compute_threshold_datapoints` returns Timestamps; `build_threshold_analysis` ISO-serialises; `build_threshold_aggregate` consumes ISO strings and re-parses inside `gaussian_smooth_with_ci` via `pd.to_datetime`.
- No placeholders; all code complete.
