# ECI Bootstrap Gap Criterion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use `epoch-research/eci-public` to make every "open has caught up to SOTA" decision for the ECI benchmark with the exact paired bootstrap (open's ECI ahead in ≥5% of paired resamples), replacing the analytical Normal approximation — across the gap-analysis chart, headline stats, and the Gap Over Time timeline. ECI only; all other benchmarks unchanged.

**Architecture:** A new `scripts/eci_bootstrap.py` refits the ECI IRT model with 500 bootstrap resamples and exposes per-model capability draws keyed by `Model` name. A single predicate in `scripts/update_data.py` (`_open_caught_up`) gains a bootstrap branch and is threaded through the four ECI gap functions plus a new `build_frontier_match_map`. The Gap Over Time chart (which matches in JS) consumes a new server-emitted `frontier_matches` map for ECI and falls back to its current JS threshold for other benchmarks. Displayed ECI scores stay Epoch-published; the fit supplies draws *only* for matching. Everything is fail-open: any failure reverts ECI to today's analytical behavior.

**Tech Stack:** Python 3.10+ (numpy, pandas, scipy, requests), the `eci` package (pip from git, pinned), pytest, vanilla JS (SVG chart).

**Key facts (validated against upstream `ae5a5db`):**
- `fit_eci_model(df, bootstrap_samples=B, bootstrap_seed=12345, use_analytical_jacobian=True, return_bootstrap_samples=True)` → `(model_df, bench_df, bootstrap)` where `bootstrap["model_names"]` is a list of `Model` names and `bootstrap["capability_samples"]` is a list (len ≤ B) of 1-D arrays of length `n_models`, aligned to `model_names`.
- ECI = positive-affine of capability, so `eci_a > eci_b ⟺ capability_a > capability_b` within a resample → compare raw capabilities, no rescaling.
- `https://epoch.ai/data/eci_benchmarks.csv` is public, has columns `model_id, benchmark_id, performance, benchmark, Model`, includes anchors Claude 3.5 Sonnet / GPT-5 / Winogrande.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/eci_bootstrap.py` | **New.** Fetch `eci_benchmarks.csv`, refit with bootstrap via `eci-public`, expose `EciBootstrap.prob_exceeds(a,b)`. Cache + fail-open. No knowledge of gaps. |
| `scripts/update_data.py` | Extend `_open_caught_up` with a bootstrap branch (+ `_match_method`); thread `bootstrap` through `calculate_horizontal_gaps`, `calculate_gap_metrics`, `calculate_statistics`, `calculate_historical_gaps`; add `build_frontier_match_map`; wire into `process_data`; emit `frontier_matches`. |
| `static/script.js` | `renderHistoricalChart`: use server `frontier_matches[framing]` for ECI, JS threshold fallback otherwise; idempotent methodology note. |
| `requirements.txt` | Pin `eci` from git. |
| `.gitignore` | Ignore `data/eci_bootstrap_cache.npz`. |
| `tests/test_eci_bootstrap.py` | **New.** Unit tests for `EciBootstrap`, `build_eci_bootstrap` fail-open + cache, and an end-to-end ECI run with synthetic draws. |
| `tests/test_gap_calculations.py` | Add bootstrap-branch tests; existing tests must stay green untouched. |
| `README.md` | Note ECI bootstrap criterion + `eci-public` attribution. |

---

### Task 1: Pin dependency and ignore the cache

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Add the pinned package to `requirements.txt`**

Append this line (keep existing lines):
```
eci @ git+https://github.com/epoch-research/eci-public.git@ae5a5db79560bd61c354202abf4d37a148335f20
```

- [ ] **Step 2: Ignore the bootstrap cache in `.gitignore`**

Append:
```
# Cached ECI bootstrap draws (regenerated from eci_benchmarks.csv)
data/eci_bootstrap_cache.npz
```

- [ ] **Step 3: Install and verify the package imports**

Run: `pip install -r requirements.txt`
Then: `python -c "import eci; from eci import fit_eci_model, load_benchmark_data; print('eci', eci.__version__)"`
Expected: prints `eci 0.1.0` with no ImportError.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "build: pin eci-public (bootstrap fitting) and ignore its cache"
```

---

### Task 2: `EciBootstrap` lookup object

**Files:**
- Create: `scripts/eci_bootstrap.py`
- Test: `tests/test_eci_bootstrap.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eci_bootstrap.py`:
```python
"""Tests for the eci-public bootstrap wrapper."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from eci_bootstrap import EciBootstrap


def _toy():
    # A: always above B, sometimes above C. B below everyone.
    draws = {
        "A": np.array([10.0, 11.0, 12.0, 9.0, 10.5]),
        "B": np.array([1.0, 2.0, 1.5, 0.5, 1.2]),
        "C": np.array([10.5, 10.0, 11.5, 9.5, 10.0]),
    }
    return EciBootstrap(draws, n_samples=5, seed=1, source_hash="h")


def test_prob_exceeds_basic():
    b = _toy()
    assert b.prob_exceeds("A", "B") == 1.0           # A > B in all 5
    # A>C in samples where 10>10.5? no; 11>10 yes; 12>11.5 yes; 9>9.5 no; 10.5>10 yes -> 3/5
    assert b.prob_exceeds("A", "C") == pytest.approx(0.6)
    assert b.prob_exceeds("B", "A") == 0.0


def test_prob_exceeds_missing_returns_none():
    b = _toy()
    assert b.prob_exceeds("A", "ZZZ") is None
    assert b.prob_exceeds("ZZZ", "A") is None


def test_has_and_model_names():
    b = _toy()
    assert b.has("A") and not b.has("ZZZ")
    assert b.model_names == {"A", "B", "C"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eci_bootstrap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eci_bootstrap'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/eci_bootstrap.py`:
```python
"""Refit the ECI IRT model with bootstrap (via eci-public) and expose paired
capability draws for the gap "caught up" criterion.

Displayed ECI scores come from Epoch's published benchmarked_models.csv; this
module supplies ONLY the joint bootstrap draws used to decide matches. Every
public entry point is fail-open: build_eci_bootstrap returns None on any
failure so the daily pipeline reverts to the analytical criterion.
"""
from __future__ import annotations

import hashlib
import logging
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ECI_BENCHMARKS_URL = "https://epoch.ai/data/eci_benchmarks.csv"
BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_SEED = 12345
CACHE_FILENAME = "eci_bootstrap_cache.npz"
# epoch.ai blocks the default urllib User-Agent with 403.
EPOCH_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQUIRED_COLUMNS = {"model_id", "benchmark_id", "performance", "benchmark", "Model"}


class EciBootstrap:
    """Joint bootstrap capability draws keyed by Epoch ``Model`` name.

    ``draws[name]`` is a 1-D array of length B; ``draws[a][i]`` and
    ``draws[b][i]`` are the same resample i, so comparisons are paired.
    """

    def __init__(self, draws: dict[str, np.ndarray], *, n_samples: int,
                 seed: int, source_hash: str):
        self.draws = draws
        self.n_samples = n_samples
        self.seed = seed
        self.source_hash = source_hash

    def has(self, name: str) -> bool:
        return name in self.draws

    @property
    def model_names(self) -> set[str]:
        return set(self.draws.keys())

    def prob_exceeds(self, a: str, b: str) -> Optional[float]:
        """P(capability_a > capability_b) across paired resamples, or None if
        either model is absent from the fit."""
        da = self.draws.get(a)
        db = self.draws.get(b)
        if da is None or db is None:
            return None
        n = min(len(da), len(db))
        if n == 0:
            return None
        return float((da[:n] > db[:n]).mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eci_bootstrap.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/eci_bootstrap.py tests/test_eci_bootstrap.py
git commit -m "feat: EciBootstrap paired-draw lookup (prob_exceeds)"
```

---

### Task 3: `build_eci_bootstrap` — fit, cache, fail-open

**Files:**
- Modify: `scripts/eci_bootstrap.py`
- Test: `tests/test_eci_bootstrap.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eci_bootstrap.py`:
```python
import eci_bootstrap as ebm


def test_build_failopen_on_fetch_error(monkeypatch):
    def boom(url, timeout=120):
        raise RuntimeError("network down")
    monkeypatch.setattr(ebm, "_fetch_csv_text", boom)
    assert ebm.build_eci_bootstrap(cache_dir=None) is None


def test_build_failopen_on_fit_error(monkeypatch):
    monkeypatch.setattr(ebm, "_fetch_csv_text",
                        lambda url, timeout=120: "model_id,benchmark_id,performance,benchmark,Model\n")
    monkeypatch.setattr(ebm, "_read_validated_df", lambda text: object())

    def boom(df, n_samples, seed, use_analytical_jacobian):
        raise ValueError("fit failed")
    monkeypatch.setattr(ebm, "_fit_capability_draws", boom)
    assert ebm.build_eci_bootstrap(cache_dir=None) is None


def test_build_and_cache_roundtrip(monkeypatch, tmp_path):
    text = "model_id,benchmark_id,performance,benchmark,Model\nm1,b1,0.5,X,A\n"
    monkeypatch.setattr(ebm, "_fetch_csv_text", lambda url, timeout=120: text)
    monkeypatch.setattr(ebm, "_read_validated_df", lambda t: "DF")

    calls = {"n": 0}

    def fake_fit(df, n_samples, seed, use_analytical_jacobian):
        calls["n"] += 1
        names = ["A", "B"]
        matrix = np.array([[3.0, 1.0], [4.0, 1.0], [2.0, 5.0]])  # (B=3, 2 models)
        return names, matrix
    monkeypatch.setattr(ebm, "_fit_capability_draws", fake_fit)

    b1 = ebm.build_eci_bootstrap(n_samples=3, cache_dir=str(tmp_path))
    assert b1 is not None
    assert calls["n"] == 1
    assert b1.prob_exceeds("A", "B") == pytest.approx(2 / 3)  # 3>1,4>1,2<5

    # Second call with same input hash loads cache, does NOT refit.
    b2 = ebm.build_eci_bootstrap(n_samples=3, cache_dir=str(tmp_path))
    assert b2 is not None
    assert calls["n"] == 1
    assert b2.prob_exceeds("A", "B") == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eci_bootstrap.py -v`
Expected: FAIL — `build_eci_bootstrap` / `_fetch_csv_text` / `_read_validated_df` / `_fit_capability_draws` not defined.

- [ ] **Step 3: Write the implementation**

Append to `scripts/eci_bootstrap.py`:
```python
def _fetch_csv_text(url: str, timeout: int = 120) -> str:
    import requests
    resp = requests.get(url, headers={"User-Agent": EPOCH_UA}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _read_validated_df(text: str):
    import pandas as pd
    df = pd.read_csv(StringIO(text))
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"eci_benchmarks.csv missing columns: {sorted(missing)}")
    return df


def _fit_capability_draws(df, n_samples: int, seed: int,
                          use_analytical_jacobian: bool):
    """Run eci-public's bootstrap fit; return (model_names, matrix[B, n_models])."""
    from eci import fit_eci_model
    _, _, bootstrap = fit_eci_model(
        df,
        bootstrap_samples=n_samples,
        bootstrap_seed=seed,
        use_analytical_jacobian=use_analytical_jacobian,
        return_bootstrap_samples=True,
    )
    model_names = list(bootstrap["model_names"])
    samples = bootstrap["capability_samples"]
    if not samples:
        raise ValueError("eci-public returned no bootstrap samples")
    matrix = np.vstack([np.asarray(s, dtype=float) for s in samples])
    if matrix.shape[1] != len(model_names):
        raise ValueError("bootstrap sample width != number of model names")
    return model_names, matrix


def _draws_from_matrix(names, matrix) -> dict[str, np.ndarray]:
    return {name: matrix[:, i] for i, name in enumerate(names)}


def _load_cache(cache_path: Path, source_hash: str, n_samples: int,
                seed: int) -> Optional["EciBootstrap"]:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=True) as data:
            if (str(data["source_hash"]) != source_hash
                    or int(data["seed"]) != seed
                    or int(data["requested_samples"]) != n_samples):
                return None
            names = [str(x) for x in data["model_names"]]
            matrix = np.asarray(data["matrix"], dtype=float)
    except Exception as e:
        logger.warning("Ignoring unreadable ECI bootstrap cache: %s", e)
        return None
    return EciBootstrap(_draws_from_matrix(names, matrix),
                        n_samples=matrix.shape[0], seed=seed,
                        source_hash=source_hash)


def _save_cache(cache_path: Path, names, matrix, source_hash: str,
                n_samples: int, seed: int) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            model_names=np.array(names, dtype=object),
            matrix=matrix,
            source_hash=source_hash,
            requested_samples=n_samples,
            seed=seed,
        )
    except Exception as e:
        logger.warning("Could not write ECI bootstrap cache: %s", e)


def build_eci_bootstrap(url: str = ECI_BENCHMARKS_URL,
                        n_samples: int = BOOTSTRAP_SAMPLES,
                        seed: int = BOOTSTRAP_SEED,
                        use_analytical_jacobian: bool = True,
                        cache_dir=None) -> Optional["EciBootstrap"]:
    """Fetch eci_benchmarks.csv, fit with bootstrap, and return paired
    capability draws keyed by ``Model``. Returns None on ANY failure so the
    daily build never breaks. ``cache_dir`` (e.g. "data") caches the matrix
    keyed by a hash of the source CSV; pass None to disable caching."""
    try:
        text = _fetch_csv_text(url)
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        cache_path = Path(cache_dir) / CACHE_FILENAME if cache_dir else None
        if cache_path is not None:
            cached = _load_cache(cache_path, source_hash, n_samples, seed)
            if cached is not None:
                logger.info("Loaded ECI bootstrap from cache (%d models, B=%d)",
                            len(cached.draws), cached.n_samples)
                return cached

        df = _read_validated_df(text)
        names, matrix = _fit_capability_draws(df, n_samples, seed,
                                              use_analytical_jacobian)
        boot = EciBootstrap(_draws_from_matrix(names, matrix),
                            n_samples=matrix.shape[0], seed=seed,
                            source_hash=source_hash)
        if cache_path is not None:
            _save_cache(cache_path, names, matrix, source_hash, n_samples, seed)
        logger.info("Fitted ECI bootstrap: %d models, B=%d",
                    len(boot.draws), boot.n_samples)
        return boot
    except Exception as e:
        logger.warning(
            "ECI bootstrap unavailable, falling back to analytical criterion: %s", e)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eci_bootstrap.py -v`
Expected: all passed (6 total).

- [ ] **Step 5: Commit**

```bash
git add scripts/eci_bootstrap.py tests/test_eci_bootstrap.py
git commit -m "feat: build_eci_bootstrap with cache and fail-open fallback"
```

---

### Task 4: Bootstrap branch in `_open_caught_up`

**Files:**
- Modify: `scripts/update_data.py` (import block near line 58-84; `_open_caught_up` near line 706-756)
- Test: `tests/test_gap_calculations.py`

- [ ] **Step 1: Write the failing tests**

Append a new class to `tests/test_gap_calculations.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gap_calculations.py::TestBootstrapCriterion -v`
Expected: FAIL — cannot import `_match_method` / `CAUGHT_UP_PROB`.

- [ ] **Step 3: Add the import + constants + refactor the predicate**

In `scripts/update_data.py`, after the METR fetcher import block (around line 84), add:
```python
# ECI bootstrap (eci-public). Lazy: importing this module never imports the
# heavy `eci` package, so update_data always loads even without it installed.
try:
    from eci_bootstrap import build_eci_bootstrap, EciBootstrap
    ECI_BOOTSTRAP_AVAILABLE = True
except Exception as e:  # pragma: no cover - defensive
    build_eci_bootstrap = None
    EciBootstrap = None
    ECI_BOOTSTRAP_AVAILABLE = False
    print(f"Note: ECI bootstrap module unavailable ({e})")

# Open model is "caught up" when its bootstrapped ECI exceeds the SOTA model's
# in at least this fraction of paired resamples (Epoch AI's 5% rule).
CAUGHT_UP_PROB = 0.05
```

Replace the body of `_open_caught_up` (the `if (...): se = ...; return ...` / `return open_score >= sota_score - threshold` tail) and add `_match_method` just above it:
```python
def _match_method(open_std, sota_std, *, open_name=None, sota_name=None,
                  bootstrap=None) -> str:
    """Which criterion applies to this pair: 'bootstrap' (paired draws exist
    for both models), 'analytical' (usable std on both), or 'threshold'."""
    if (bootstrap is not None and open_name and sota_name
            and bootstrap.prob_exceeds(open_name, sota_name) is not None):
        return "bootstrap"
    if pd.notna(open_std) and pd.notna(sota_std) and (open_std > 0 or sota_std > 0):
        return "analytical"
    return "threshold"


def _open_caught_up(
    open_score: float,
    open_std: float,
    sota_score: float,
    sota_std: float,
    threshold: float,
    z: float = Z_ONE_SIDED_05,
    *,
    open_name: Optional[str] = None,
    sota_name: Optional[str] = None,
    bootstrap=None,
) -> bool:
    """Has the open model *plausibly caught up* to a historical SOTA model?

    Preferred path (ECI, when joint bootstrap draws exist for both models):
    the exact paired-bootstrap test — caught up iff the open model's
    bootstrapped ECI exceeds the SOTA model's in >= CAUGHT_UP_PROB of paired
    resamples. This is Epoch AI's published criterion computed directly,
    without the analytical independence assumption.

    Fallbacks (no draws): the analytical Normal mirror
    ``(sota - open) <= z * sqrt(s_open^2 + s_sota^2)`` when both std are usable,
    then a point-estimate match within ``threshold``. These two fallbacks
    reproduce the prior behavior exactly.
    """
    method = _match_method(open_std, sota_std, open_name=open_name,
                           sota_name=sota_name, bootstrap=bootstrap)
    if method == "bootstrap":
        return bootstrap.prob_exceeds(open_name, sota_name) >= CAUGHT_UP_PROB
    if method == "analytical":
        se = math.sqrt(open_std ** 2 + sota_std ** 2)
        return (sota_score - open_score) <= z * se
    return open_score >= sota_score - threshold
```

Keep `Z_ONE_SIDED_05` defined above `_open_caught_up` (it already is). `Optional` is already imported (`from typing import ... Optional`).

- [ ] **Step 4: Run tests to verify they pass (new + all existing predicate tests)**

Run: `pytest tests/test_gap_calculations.py -v`
Expected: all passed, including the existing `TestOpenCaughtUp` (unchanged behavior) and the new `TestBootstrapCriterion`.

- [ ] **Step 5: Commit**

```bash
git add scripts/update_data.py tests/test_gap_calculations.py
git commit -m "feat: paired-bootstrap branch in _open_caught_up (+ _match_method)"
```

---

### Task 5: Thread bootstrap into `calculate_horizontal_gaps`

**Files:**
- Modify: `scripts/update_data.py` (`calculate_horizontal_gaps`, ~line 259-335)
- Test: `tests/test_gap_calculations.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gap_calculations.py`:
```python
class TestHorizontalGapsBootstrap:
    def test_bootstrap_match_more_permissive_than_threshold(self):
        # Open 98.5 vs Closed 100: threshold(1.0) says NOT matched (98.5<99).
        # Bootstrap with P(open>closed)=0.10 says matched.
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [100.0, 98.5],
            "eci_std": [1.0, 1.0],
            "date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "Open": [False, True],
        })
        a = _np.concatenate([_np.full(10, 200.0), _np.zeros(90)])  # P(open>closed)=0.10
        b = _np.full(100, 100.0)
        boot = EciBootstrap({"OpenB": a, "ClosedA": b}, n_samples=100, seed=1, source_hash="h")

        no_boot = calculate_horizontal_gaps(df)
        assert no_boot[0]["matched"] is False  # threshold path

        with_boot = calculate_horizontal_gaps(df, bootstrap=boot)
        assert with_boot[0]["matched"] is True
        assert with_boot[0]["match_type"] == "bootstrap"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gap_calculations.py::TestHorizontalGapsBootstrap -v`
Expected: FAIL — `calculate_horizontal_gaps` has no `bootstrap` kwarg.

- [ ] **Step 3: Implement**

Change the signature and matching loop of `calculate_horizontal_gaps`. New signature:
```python
def calculate_horizontal_gaps(
    df: pd.DataFrame,
    score_col: str = "eci",
    threshold: float = ECI_MATCH_THRESHOLD,
    model_col: str = "Model",
    bootstrap=None,
) -> list[dict]:
```
Add near the top of the function body (after the empty-df guard):
```python
    std_col = f"{score_col}_std"
```
Replace the inner match test block:
```python
            # Match if open model is within threshold of closed model
            if open_row[score_col] >= closed_score - threshold:
                matching_open = open_row
                match_type = "exact"
                break
```
with:
```python
            open_name = open_row.get(model_col, open_row.get("model"))
            closed_name = closed_row.get(model_col, closed_row.get("model"))
            if _open_caught_up(
                open_row[score_col], open_row.get(std_col, np.nan),
                closed_score, closed_row.get(std_col, np.nan),
                threshold,
                open_name=open_name, sota_name=closed_name, bootstrap=bootstrap,
            ):
                matching_open = open_row
                match_type = _match_method(
                    open_row.get(std_col, np.nan), closed_row.get(std_col, np.nan),
                    open_name=open_name, sota_name=closed_name, bootstrap=bootstrap,
                )
                break
```

- [ ] **Step 4: Run tests to verify they pass (new + all existing horizontal-gap tests)**

Run: `pytest tests/test_gap_calculations.py -v`
Expected: all passed. (Existing `TestHorizontalGapCalculation` dataframes have no `eci_std` → threshold path → identical behavior.)

- [ ] **Step 5: Commit**

```bash
git add scripts/update_data.py tests/test_gap_calculations.py
git commit -m "feat: bootstrap criterion in calculate_horizontal_gaps (ECI)"
```

---

### Task 6: Thread bootstrap into `calculate_gap_metrics` and `calculate_statistics`

**Files:**
- Modify: `scripts/update_data.py` (`calculate_gap_metrics` ~759-902; `calculate_statistics` ~905-1005)
- Test: `tests/test_gap_calculations.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gap_calculations.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gap_calculations.py::TestGapMetricsBootstrap -v`
Expected: FAIL — `calculate_gap_metrics` / `calculate_statistics` have no `bootstrap` kwarg.

- [ ] **Step 3: Implement**

In `calculate_gap_metrics`, add `bootstrap=None` to the signature (after `z`). Where the `sota` list is built, add a `name` field:
```python
            sota.append({
                "date": r["date"],
                "score": float(s),
                "std": float(r[std_col]) if has_std and pd.notna(r.get(std_col)) else np.nan,
                "name": r.get("Model", r.get("model")),
            })
```
Capture the best-open name just after computing `best_open`:
```python
        best_open_name = best_open.get("Model", best_open.get("model"))
```
In the lenient time-gap loop, pass names + bootstrap:
```python
        ref_date = None
        for s in sorted(sota_avail, key=lambda x: x["date"], reverse=True):
            if _open_caught_up(best_open_score, best_open_std, s["score"], s["std"],
                               threshold, z, open_name=best_open_name,
                               sota_name=s["name"], bootstrap=bootstrap):
                ref_date = s["date"]
                break
```
(The strict loop stays a point-estimate comparison — do **not** change it.)

In `calculate_statistics`, add `bootstrap=None` to the signature and forward it:
```python
    metrics = calculate_gap_metrics(
        df,
        score_col=score_col,
        threshold=threshold,
        window_start=window_start,
        window_end=window_end,
        bootstrap=bootstrap,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gap_calculations.py -v`
Expected: all passed (existing `TestGapMetrics` / `TestStatistics*` unaffected — no bootstrap passed there).

- [ ] **Step 5: Commit**

```bash
git add scripts/update_data.py tests/test_gap_calculations.py
git commit -m "feat: bootstrap criterion in calculate_gap_metrics/statistics (ECI)"
```

---

### Task 7: Thread bootstrap into `calculate_historical_gaps`

**Files:**
- Modify: `scripts/update_data.py` (`calculate_historical_gaps` ~613-697)
- Test: `tests/test_gap_calculations.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gap_calculations.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gap_calculations.py::TestHistoricalGapsBootstrap -v`
Expected: FAIL — no `bootstrap` kwarg.

- [ ] **Step 3: Implement**

Change the signature:
```python
def calculate_historical_gaps(
    df: pd.DataFrame,
    score_col: str = "eci",
    threshold: float = ECI_MATCH_THRESHOLD,
    model_col: str = "Model",
    bootstrap=None,
) -> list[dict]:
```
Add `std_col = f"{score_col}_std"` after the `df = df.sort_values("date")` line. Inside the monthly loop, after `best_open` / `best_closed` are computed, capture names/std and replace the two threshold comparisons:

Replace:
```python
        # Find the first closed model to achieve the best open's score level
        closed_at_open_level = df_closed[df_closed[score_col] >= best_open_score - threshold].sort_values("date")

        if len(closed_at_open_level) > 0:
            first_closed_at_level = closed_at_open_level.iloc[0]
```
with:
```python
        best_open_name = best_open.get(model_col, best_open.get("model"))
        best_open_std = best_open.get(std_col, np.nan)
        best_closed_name = best_closed.get(model_col, best_closed.get("model"))
        best_closed_std = best_closed.get(std_col, np.nan)

        # First closed model (by date) to reach the best open's level: the
        # leader has "caught up to" the open frontier (a=closed, b=open).
        first_closed_at_level = None
        for _, c in df_closed.sort_values("date").iterrows():
            if _open_caught_up(
                c[score_col], c.get(std_col, np.nan),
                best_open_score, best_open_std, threshold,
                open_name=c.get(model_col, c.get("model")),
                sota_name=best_open_name, bootstrap=bootstrap,
            ):
                first_closed_at_level = c
                break

        if first_closed_at_level is not None:
```
And replace the `is_matched` line:
```python
            is_matched = best_open_score >= best_closed_score - threshold
```
with:
```python
            is_matched = _open_caught_up(
                best_open_score, best_open_std,
                best_closed_score, best_closed_std, threshold,
                open_name=best_open_name, sota_name=best_closed_name,
                bootstrap=bootstrap,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gap_calculations.py -v`
Expected: all passed (existing `TestHistoricalGaps` has no std → threshold path → unchanged; gap still clamped `max(0, ...)`).

- [ ] **Step 5: Commit**

```bash
git add scripts/update_data.py tests/test_gap_calculations.py
git commit -m "feat: bootstrap criterion in calculate_historical_gaps (ECI)"
```

---

### Task 8: `build_frontier_match_map` + wire bootstrap into `process_data`

**Files:**
- Modify: `scripts/update_data.py` (`process_data` ~1066-1169; `process_all_benchmarks` eci payload ~1433-1449; add new helper)
- Test: `tests/test_gap_calculations.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gap_calculations.py`:
```python
from update_data import build_frontier_match_map


class TestFrontierMatchMap:
    def test_maps_each_laggard_to_earliest_caught_up_leader(self):
        # Open frontier: O1(2024-06). Closed frontier: C_old(2023-01,100), C_new(2024-01,120).
        df = pd.DataFrame({
            "Model": ["C_old", "C_new", "O1"],
            "eci": [100.0, 120.0, 119.0],
            "eci_std": [2.0, 2.0, 2.0],
            "date": pd.to_datetime(["2023-01-01", "2024-01-01", "2024-06-01"]),
            "Open": [False, False, True],
        })
        # caught_up(leader, laggard): C_old plausibly >= O1? P(C_old>O1)=0 -> no.
        # C_new plausibly >= O1? P(C_new>O1)=0.30 -> yes. Earliest such leader = C_new.
        boot = EciBootstrap({
            "O1": _np.full(100, 119.0),
            "C_old": _np.zeros(100),
            "C_new": _np.concatenate([_np.full(30, 999.0), _np.full(70, 0.0)]),
        }, n_samples=100, seed=1, source_hash="h")
        m = build_frontier_match_map(df, boot)
        assert m == {"O1": "C_new"}

    def test_threshold_fallback_without_bootstrap(self):
        df = pd.DataFrame({
            "Model": ["C_old", "C_new", "O1"],
            "eci": [100.0, 120.0, 119.0],
            "date": pd.to_datetime(["2023-01-01", "2024-01-01", "2024-06-01"]),
            "Open": [False, False, True],
        })
        # threshold 1.0: earliest leader with score >= 119-1=118 -> C_new(120).
        m = build_frontier_match_map(df, None)
        assert m == {"O1": "C_new"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gap_calculations.py::TestFrontierMatchMap -v`
Expected: FAIL — `build_frontier_match_map` not defined.

- [ ] **Step 3: Implement the helper**

Add to `scripts/update_data.py` (near `calculate_historical_gaps`):
```python
def build_frontier_match_map(
    df_frontier: pd.DataFrame,
    bootstrap=None,
    score_col: str = "eci",
    model_col: str = "Model",
    threshold: float = ECI_MATCH_THRESHOLD,
    laggard_col: str = "Open",
) -> dict:
    """For each laggard running-max frontier model, the earliest leader
    running-max frontier model it has reached — i.e. the earliest leader L with
    ``_open_caught_up(open=L, sota=laggard)``. Mirrors the JS firstLeaderToReach
    but uses the bootstrap predicate when draws exist. Used to drive the Gap
    Over Time chart server-side. Keys/values are ``model_col`` names.

    Works for both framings: pass the open/closed frontier (laggard = Open) or
    the China/US frontier (process_data sets ``Open`` = is_china there)."""
    std_col = f"{score_col}_std"
    d = df_frontier.dropna(subset=["date", score_col]).sort_values(
        "date", kind="mergesort")

    laggards, leaders = [], []
    run_lag = run_led = -np.inf
    for _, r in d.iterrows():
        s = float(r[score_col])
        if bool(r[laggard_col]):
            if s > run_lag:
                run_lag = s
                laggards.append(r)
        else:
            if s > run_led:
                run_led = s
                leaders.append(r)

    result: dict = {}
    for lag in laggards:
        lag_name = lag.get(model_col, lag.get("model"))
        lag_std = lag.get(std_col, np.nan)
        matched = None
        for ld in leaders:  # ascending by date
            if _open_caught_up(
                ld[score_col], ld.get(std_col, np.nan),
                lag[score_col], lag_std, threshold,
                open_name=ld.get(model_col, ld.get("model")),
                sota_name=lag_name, bootstrap=bootstrap,
            ):
                matched = ld.get(model_col, ld.get("model"))
                break
        result[lag_name] = matched
    return result
```

- [ ] **Step 4: Run the helper tests**

Run: `pytest tests/test_gap_calculations.py::TestFrontierMatchMap -v`
Expected: 2 passed.

- [ ] **Step 5: Wire bootstrap + frontier_matches into `process_data`**

In `process_data`, right after `df_frontier` is created (after line ~1081), add:
```python
    bootstrap = build_eci_bootstrap() if build_eci_bootstrap is not None else None
    if bootstrap is not None:
        displayed = {m.get("Model") for _, m in df_frontier.iterrows()}
        matched = sum(1 for n in displayed if n and bootstrap.has(n))
        logger.info("ECI bootstrap name-join: %d/%d frontier models matched",
                    matched, len(displayed))
```
Pass `bootstrap=bootstrap` to the open/closed calls:
```python
    gaps = calculate_horizontal_gaps(df_frontier, score_col="eci", threshold=ECI_MATCH_THRESHOLD, model_col="Model", bootstrap=bootstrap)
    ...
    stats = calculate_statistics(df_frontier, gaps, score_col="eci", bootstrap=bootstrap)
    ...
    historical_gaps = calculate_historical_gaps(df_frontier, score_col="eci", threshold=ECI_MATCH_THRESHOLD, model_col="Model", bootstrap=bootstrap)
```
Pass `bootstrap=bootstrap` to the three China-framing calls as well:
```python
        china_gaps = calculate_horizontal_gaps(df_china_us_frontier, score_col="eci", threshold=ECI_MATCH_THRESHOLD, model_col="Model", bootstrap=bootstrap)
        ...
        china_stats = calculate_statistics(df_china_us_frontier, china_gaps, score_col="eci", bootstrap=bootstrap)
        ...
        china_historical = calculate_historical_gaps(df_china_us_frontier, score_col="eci", threshold=ECI_MATCH_THRESHOLD, model_col="Model", bootstrap=bootstrap)
```
Build the frontier-match maps. After the China block, before `return`:
```python
    frontier_matches = {"default": build_frontier_match_map(df_frontier, bootstrap)}
    if len(df_china_models) > 0 and len(df_us_models) > 0:
        frontier_matches["china"] = build_frontier_match_map(df_china_us_frontier, bootstrap)
```
Add `"frontier_matches": frontier_matches,` to the returned dict (alongside `"china_framing"`).

In `process_all_benchmarks`, add to the `benchmarks["eci"]` dict:
```python
        "frontier_matches": eci_data["frontier_matches"],
```

- [ ] **Step 6: Verify the full suite still passes**

Run: `pytest tests/ -v`
Expected: all passed.

- [ ] **Step 7: Commit**

```bash
git add scripts/update_data.py tests/test_gap_calculations.py
git commit -m "feat: emit ECI frontier_matches and wire bootstrap through process_data"
```

---

### Task 9: Frontend — Gap Over Time uses server matches for ECI

**Files:**
- Modify: `static/script.js` (`renderHistoricalChart`, ~line 1487-1547)

- [ ] **Step 1: Add server-match lookup + methodology note**

In `renderHistoricalChart`, after `benchmark` and `laggardFrontierRaw`/`leaderFrontier` are defined (after line ~1522), and before the `firstLeaderToReach` block, insert:
```javascript
    // ECI ships a server-computed bootstrap match map; other benchmarks match
    // client-side via threshold. Map framing -> server key.
    const fkey = framing === 'china' ? 'china' : 'default';
    const serverMatches = benchmark?.frontier_matches?.[fkey] || null;

    // idempotent methodology note for the bootstrap-matched (ECI) view
    let methodNote = document.getElementById('historical-method-note');
    if (serverMatches) {
        if (!methodNote && container) {
            methodNote = document.createElement('p');
            methodNote.id = 'historical-method-note';
            methodNote.className = 'chart-note';
            container.appendChild(methodNote);
        }
        if (methodNote) {
            methodNote.textContent =
                'Matches use Epoch AI’s paired bootstrap (open model ahead in ≥5% of resamples). ' +
                'Plotted scores are Epoch’s published ECI point estimates.';
            methodNote.hidden = false;
        }
    } else if (methodNote) {
        methodNote.hidden = true;
    }

    function matchedLeaderFor(ev) {
        if (serverMatches) {
            const lm = serverMatches[ev.model];
            return lm ? (leaderFrontier.find(l => l.model === lm) || null) : null;
        }
        return firstLeaderToReach(ev.score);
    }
```
Then change the laggard loop to use it. Replace:
```javascript
    for (const ev of laggardFrontierRaw) {
        const ld = firstLeaderToReach(ev.score);
```
with:
```javascript
    for (const ev of laggardFrontierRaw) {
        const ld = matchedLeaderFor(ev);
```
(Leave `firstLeaderToReach`'s definition in place — it is the fallback path.)

- [ ] **Step 2: Verify HTML/structure test still passes**

Run: `pytest tests/test_index_html_structure.py -v`
Expected: all passed (no structural change to `index.html`).

- [ ] **Step 3: Manual smoke check (optional, after data regen in Task 10)**

After `data.json` is regenerated, open `python app.py` → http://localhost:8080, switch to ECI, confirm the Gap Over Time chart renders and shows the methodology note; switch to another benchmark and confirm the note disappears and the chart still renders.

- [ ] **Step 4: Commit**

```bash
git add static/script.js
git commit -m "feat: Gap Over Time uses server bootstrap matches for ECI"
```

---

### Task 10: End-to-end validation, docs, senior review

**Files:**
- Modify: `README.md`
- (validation only) `scripts/update_data.py`, `scripts/eci_bootstrap.py`

- [ ] **Step 1: Fast end-to-end smoke run (small B)**

Run a throwaway check that the real pipeline produces `data.json` with the new key, using a small bootstrap for speed:
```bash
python -c "import sys; sys.path.insert(0,'scripts'); import eci_bootstrap as e; b=e.build_eci_bootstrap(n_samples=20, cache_dir=None); print('boot models:', None if b is None else len(b.draws))"
```
Expected: prints `boot models: <~164>` (or `None` only if offline — then note it and rely on CI).

- [ ] **Step 2: Full regeneration (real B=500) and assertions**

Run: `python scripts/update_data.py`
Then verify:
```bash
python -c "import json; d=json.load(open('data.json')); eci=d['benchmarks']['eci']; print('has frontier_matches:', 'frontier_matches' in eci); print('framings:', list(eci['frontier_matches'].keys())); print('avg gap months:', eci['statistics']['avg_horizontal_gap_months'])"
```
Expected: `has frontier_matches: True`, framings include `default` (and `china` if data present), and a sensible gap value. (This step exercises the real fit; expect a multi-minute run with the tqdm bootstrap bar.)

- [ ] **Step 3: Update README**

In `README.md`, under the Benchmarks list / Data Source section, add a short paragraph:
```markdown
### ECI Catch-Up Criterion

For the **Epoch Capabilities Index (ECI)** only, "has an open model caught up to
a prior state-of-the-art closed model?" is decided with Epoch AI's paired
bootstrap: we refit the ECI model on bootstrap resamples using
[`eci-public`](https://github.com/epoch-research/eci-public) and count a model
as caught up when its ECI exceeds the reference model's in at least 5% of paired
resamples. Displayed ECI scores remain Epoch's published point estimates; the
bootstrap is used only for the matching decision. If the refit is unavailable,
the app falls back to an analytical approximation. All other benchmarks use a
fixed point-estimate threshold.
```

- [ ] **Step 4: Senior review (per CLAUDE.md)**

Run:
```bash
gemini -p "@scripts/eci_bootstrap.py You are a Senior Software Engineer. Review this code for: data handling errors, off-by-one bugs, incorrect aggregations, missing edge cases, error recovery, and credential/security concerns."
gemini -p "@scripts/update_data.py Review the ECI bootstrap criterion changes for: logical correctness of the paired-bootstrap predicate, direction of the open-vs-SOTA comparison, edge cases when bootstrap draws are missing, and data leakage between benchmarks."
```
Implement any correct suggestions immediately; re-run `pytest tests/ -v`.

- [ ] **Step 5: Final commit**

```bash
git add README.md
git commit -m "docs: explain ECI paired-bootstrap catch-up criterion"
```

---

## Self-Review

**Spec coverage:**
- §2 refit-in-house for draws → Task 3 (`_fit_capability_draws` via `eci-public`). ✓
- §2 display stays Epoch-published → no change to `fetch_eci_data`/model lists; only matching changed. ✓
- §2 criterion across all ECI decisions → Tasks 5 (horizontal), 6 (gap_metrics/stats), 7 (historical), 8 (frontier_matches), 9 (frontend). ✓
- §2 integration = pinned pip-from-git → Task 1. ✓
- §2 B=500 → `BOOTSTRAP_SAMPLES=500` (Task 3); validated in Task 10. ✓
- §3 scale-invariance (compare capabilities) → `prob_exceeds` on `capability_samples` (Tasks 2-3). ✓
- §5.1 cache + fail-open → Task 3. ✓
- §5.2 predicate keeps signature/fallbacks → Task 4 (existing `TestOpenCaughtUp` unchanged). ✓
- §5.3 threading → Tasks 5-7. ✓
- §5.4 process_data orchestration + match-rate log → Task 8. ✓
- §5.5 frontier_matches {default, china} → Task 8. ✓
- §5.6 frontend match-map + note → Task 9. ✓
- §5.7 requirements pin → Task 1. ✓
- §7 fail-open end-to-end → Task 3 (None) + Task 8 (`build_eci_bootstrap() if ... else None`) + Task 9 (note hidden, JS fallback). ✓
- §8 tests + gemini review → every task is TDD; Task 10 review. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type/name consistency:** `EciBootstrap.prob_exceeds`, `_match_method`, `build_eci_bootstrap`, `build_frontier_match_map`, `CAUGHT_UP_PROB`, `frontier_matches[fkey]`, `bootstrap=` kwarg — names identical across Tasks 2-9. `bootstrap` is duck-typed (anything with `prob_exceeds`); tests pass real `EciBootstrap`. ✓

**Known behavior change (intentional):** For real ECI data with `eci_std` present but bootstrap unavailable, `calculate_horizontal_gaps`/`calculate_historical_gaps` now use the *analytical* predicate instead of the old pure threshold, making the chart consistent with the headline stat (which already used analytical). Existing tests pass because their fixtures omit `eci_std` (→ threshold path). Non-ECI benchmarks always pass `bootstrap=None` and (typically) lack usable std → threshold path → unchanged.
