# ECI Bootstrap Gap Criterion — Design

**Date:** 2026-06-02
**Status:** Approved for planning
**Scope:** ECI benchmark only. METR, GPQA Diamond, MATH Level 5, OTIS Mock AIME,
SWE-Bench Verified, SimpleQA Verified, FrontierMath, and Chess Puzzles are
explicitly untouched.

## 1. Problem

The gap analysis decides when an open-weight model has "caught up to" a prior
state-of-the-art (SOTA) closed-weight model. Epoch AI's published methodology
makes this decision from a **paired bootstrap**: it resamples the underlying
benchmark scores with replacement, refits the ECI model on each resample
(preserving the pairing between models across resamples), and counts a prior
SOTA model as *not significantly better* than the open model when the open
model's bootstrapped ECI exceeds the SOTA model's in **at least 5% of paired
samples**.

This repo currently has only Epoch's published point estimate + 90% CI per
model (from `benchmarked_models.csv`), not the joint bootstrap draws. So
`scripts/update_data.py::_open_caught_up` *mirrors* the test analytically:
treat each model's bootstrap ECI as `Normal(score, std^2)`, recover `std` from
the 90% CI, and test `P(open > sota) >= 0.05  <=>  sota - open <= z*SE`
(`z = 1.645`, `SE = sqrt(s_open^2 + s_sota^2)`). Its own docstring flags the
limitation: the analytical mirror assumes **independent marginals**, which
overstates `SE` relative to Epoch's positively-correlated paired bootstrap,
making the test marginally more permissive than the truth.

The [`eci-public`](https://github.com/epoch-research/eci-public) package is the
code that *produces* those joint bootstrap draws. Integrating it lets us run the
**exact** paired-bootstrap test instead of the analytical approximation.

## 2. Decisions (locked)

| Decision | Choice |
|---|---|
| What `eci-public` enables | Refit ECI in-house to obtain joint bootstrap draws **and** use them for the caught-up criterion. |
| Displayed ECI values (dots, leaderboard, vertical gap) | **Unchanged** — keep Epoch's published `eci` / `eci_ci_low` / `eci_ci_high` from `benchmarked_models.csv`. The in-house fit supplies bootstrap draws **only** for the criterion. |
| Criterion scope | **All** ECI matching decisions: gap-analysis chart + headline stats, *and* the Gap Over Time timeline. |
| Integration mechanism | `pip install` from git, **pinned to a commit SHA**. |
| Bootstrap resamples `B` | **500** (most faithful 5% tail; analytical Jacobian). |

## 3. Key mathematical facts (validated)

1. **`eci-public` API** (commit `ae5a5db79560bd61c354202abf4d37a148335f20`,
   2026-05-28):
   ```python
   from eci import load_benchmark_data, fit_eci_model
   df = load_benchmark_data("https://epoch.ai/data/eci_benchmarks.csv")
   model_df, bench_df, bootstrap = fit_eci_model(
       df, bootstrap_samples=500, bootstrap_seed=12345,
       use_analytical_jacobian=True, return_bootstrap_samples=True,
   )
   ```
   `bootstrap` is a dict with keys `model_ids`, `model_names`, `benchmark_ids`,
   `benchmark_names`, `capability_samples`, `difficulty_samples`,
   `discriminability_samples`. `capability_samples` is a list of length up to
   `B` of 1-D numpy arrays; `capability_samples[b][i]` is the fitted capability
   of `model_names[i]` in resample `b`.

2. **Scale invariance.** `compute_eci_scores` maps capability to ECI by a
   *positive* affine transform `eci = a + b*capability` with `b > 0` (because
   `anchor_eci_high > anchor_eci_low` and `cap_high > cap_low`). Within a single
   resample, all models share the same `a, b`, so
   `eci_a > eci_b  <=>  capability_a > capability_b`. The paired comparison is
   therefore **scale-invariant**: we compute it directly on `capability_samples`
   and never need to ECI-scale the draws. (We still only ever compare two models
   *within the same resample b*, which is exactly the paired bootstrap.)

3. **Data availability.** `https://epoch.ai/data/eci_benchmarks.csv` is public
   (HTTP 200, ~260 KB): 1,615 rows, 164 models, 42 benchmarks. It already ships
   the columns `load_benchmark_data` validates (`model_id`, `benchmark_id`,
   `performance`, `benchmark`, `Model`). Anchor models **Claude 3.5 Sonnet** and
   **GPT-5** and anchor benchmark **Winogrande** are all present, so the fit
   runs unattended with no credentials. The in-house fit covers 164 models —
   a superset of the ~140 displayed today.

## 4. The caught-up criterion

Define the low-level paired probability and the predicate:

```
P(a > b)        = mean over resamples of [ capability_a[resample] > capability_b[resample] ]
caught_up(a, b) = ( P(a > b) >= 0.05 )      # a is "not significantly worse than" b
```

`caught_up(a, b)` reads as "**a has plausibly caught up to b**." It is used in
two argument roles, matching the existing code's geometry:

- **Gap analysis** (an open model reaching a closed SOTA model):
  `caught_up(open, closed)`.
- **Gap Over Time** (the earliest closed *leader* to reach an open *laggard*'s
  level — today's `firstLeaderToReach`): `caught_up(leader, laggard)`, i.e.
  `caught_up(closed, open)`.

This is the exact, non-Gaussian, paired generalization of the current
analytical test (`P(open > sota) >= 0.05`), with the independence caveat
removed.

## 5. Components

### 5.1 `scripts/eci_bootstrap.py` (new)

A thin wrapper around `eci-public` plus a lookup object.

```python
ECI_BENCHMARKS_URL = "https://epoch.ai/data/eci_benchmarks.csv"
BOOTSTRAP_SAMPLES  = 500
BOOTSTRAP_SEED     = 12345
CAUGHT_UP_PROB     = 0.05

class EciBootstrap:
    """Joint bootstrap capability draws keyed by Epoch `Model` name."""
    def __init__(self, draws: dict[str, np.ndarray], *, n_samples: int,
                 seed: int, source_hash: str): ...
    def has(self, name: str) -> bool: ...
    def prob_exceeds(self, a: str, b: str) -> float | None:
        """P(capability_a > capability_b) across paired resamples.
        Returns None if either model is absent from the fit."""
    @property
    def model_names(self) -> set[str]: ...

def build_eci_bootstrap(
    url: str = ECI_BENCHMARKS_URL,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    use_analytical_jacobian: bool = True,
    cache_dir: str | Path | None = "data",
) -> EciBootstrap | None:
    """Fetch eci_benchmarks.csv, fit with bootstrap, return draws keyed by
    `Model`. Returns None on ANY failure (network, schema drift, missing
    anchor, import error) so the daily build never breaks."""
```

Behavior:
- Hash the fetched CSV bytes. If `cache_dir/eci_bootstrap_cache.npz` exists with
  a matching `source_hash` + `n_samples` + `seed`, load it instead of refitting.
  Otherwise fit, then write the cache. Cache file is **gitignored**.
- Build the `(B, n_models)` matrix from `capability_samples` (stack the list;
  tolerate `len < B` if some resamples were dropped) and slice columns by
  `model_names` into `draws[name] -> np.ndarray(shape=(B,))`.
- `prob_exceeds` returns `float((draws[a] > draws[b]).mean())`, or `None` if a
  name is missing.
- Wrap the whole thing in try/except; log a single WARNING and return `None` on
  failure. Log `n_samples`, `n_models`, and (set later, see 5.3) the displayed
  match rate at INFO on success.

### 5.2 Predicate change in `scripts/update_data.py`

Extend `_open_caught_up` with optional keyword args; default behavior unchanged.

```python
def _open_caught_up(open_score, open_std, sota_score, sota_std, threshold,
                    z=Z_ONE_SIDED_05, *, open_name=None, sota_name=None,
                    bootstrap=None):
    if bootstrap is not None and open_name and sota_name:
        p = bootstrap.prob_exceeds(open_name, sota_name)
        if p is not None:
            return p >= CAUGHT_UP_PROB           # exact paired bootstrap
    # --- unchanged below: analytical Normal mirror, then threshold fallback ---
```

Because the positional signature and the no-bootstrap path are preserved,
**every existing test in `tests/test_gap_calculations.py` still passes**, and
all non-ECI benchmarks (which never pass `bootstrap`) are byte-for-byte
identical.

### 5.3 Threading `bootstrap` through ECI functions

`bootstrap: Optional[EciBootstrap] = None` is added (defaulting to `None`) to:
`calculate_gap_metrics`, `calculate_horizontal_gaps`, `calculate_historical_gaps`,
and `calculate_statistics`. Each passes model names into `_open_caught_up`:

- `calculate_gap_metrics`: the `sota` list entries gain a `name` field; the
  daily lenient loop calls `_open_caught_up(best_open_score, best_open_std,
  s["score"], s["std"], threshold, z, open_name=best_open_name,
  sota_name=s["name"], bootstrap=bootstrap)`. The **strict** variant stays a
  point-estimate comparison (`best_open_score > s["score"]`) and is unaffected.
- `calculate_horizontal_gaps`: the per-closed-model match test becomes
  `_open_caught_up(open_eci, open_std, closed_eci, closed_std, threshold,
  open_name=open_Model, sota_name=closed_Model, bootstrap=bootstrap)` instead of
  the bare `open_row[score_col] >= closed_score - threshold`. `match_type` is set
  to `"bootstrap"`, `"analytical"`, or `"threshold"` to record which path fired.
- `calculate_historical_gaps`: the "first closed model at the open frontier's
  level" search and the `is_matched` flag use the same predicate with
  `a=closed, b=open` (leader reaching laggard).

Only the ECI call sites (in `process_data`) pass a non-`None` bootstrap;
`process_benchmark_data` / `process_metr_data` keep passing `None`.

### 5.4 `process_data` (ECI orchestration)

1. `bootstrap = build_eci_bootstrap()` once, near the top.
2. Pass `bootstrap=bootstrap` to the open/closed `calculate_horizontal_gaps`,
   `calculate_statistics`, `calculate_historical_gaps`, **and** the China-framing
   equivalents (the bootstrap draws are per-model and framing-agnostic; only the
   laggard/leader partition differs).
3. Build `frontier_matches` (see 5.5) for both framings.
4. Log the displayed match rate: fraction of displayed frontier models that were
   found in `bootstrap.model_names`.

### 5.5 `frontier_matches` for the Gap Over Time chart

New helper:

```python
def build_frontier_match_map(df_frontier, bootstrap, *,
                             laggard_is_open=True) -> dict[str, str | None]:
    """For each laggard running-max frontier model, return the earliest leader
    running-max frontier model it has caught up to (caught_up(leader, laggard)),
    mirroring the JS firstLeaderToReach but with the bootstrap predicate.
    Falls back to the analytical/threshold predicate when draws are missing."""
```

`process_data` emits, inside the ECI benchmark payload:

```json
"frontier_matches": {
  "default": { "<open laggard Model>": "<closed leader Model | null>", ... },
  "china":   { "<china laggard Model>": "<us leader Model | null>", ... }
}
```

This is the only new server output. It is small (≈ one entry per frontier
laggard model). All other keys retain their existing shape; the *values* of
`gaps`, `statistics`, and `historical_gaps` simply reflect the new criterion.

### 5.6 Frontend (`static/script.js`, surgical)

Only `renderHistoricalChart` changes. Today it computes, per laggard frontier
event, `firstLeaderToReach(ev.score)` (a point-estimate threshold). New logic:

```js
const serverMatches = benchmark?.frontier_matches?.[framing] || null;
function matchedLeaderFor(ev) {
  if (serverMatches) {                          // ECI: use server bootstrap match
    const leaderModel = serverMatches[ev.model];
    return leaderModel ? leaderFrontier.find(l => l.model === leaderModel) : null;
  }
  return firstLeaderToReach(ev.score);          // all other benchmarks: unchanged
}
```

`ev._d - matchedLeader._d` still yields `gapMonths`; the step-chart geometry,
tooltips, axis logic, and "current gap" mode are untouched. The **main
gap-analysis chart needs no JS change** — it already renders from server
`data.gaps`.

A short methodology note is added near the ECI Gap Over Time chart: matches use
Epoch's paired bootstrap (≥5% of resamples), displayed scores are Epoch's
published point estimates.

### 5.7 Dependency (`requirements.txt`)

Append, pinned:
```
eci @ git+https://github.com/epoch-research/eci-public.git@ae5a5db79560bd61c354202abf4d37a148335f20
```
`eci-public` deps (`numpy`, `pandas`, `scipy`, `tqdm`) are already satisfied by
the existing requirements (only `tqdm` is newly pulled in transitively). No
change to `.github/workflows/daily_update.yml` — the existing
`pip install -r requirements.txt` step installs it.

## 6. Data flow

```
eci_benchmarks.csv ──load_benchmark_data──▶ fit_eci_model(B=500, return_bootstrap_samples=True)
                                                   │
                                          capability_samples (B × 164)
                                                   │  keyed by Model name
                                                   ▼
                                            EciBootstrap.draws
                                                   │
benchmarked_models.csv ──(published eci/CI, display)──┐         │
                                                      ▼         ▼
                          process_data(): df_frontier + bootstrap
                            ├─ calculate_horizontal_gaps  → data.gaps         (main chart)
                            ├─ calculate_gap_metrics       → data.statistics  (headline)
                            ├─ calculate_historical_gaps   → data.historical_gaps
                            └─ build_frontier_match_map    → data.frontier_matches (Gap Over Time)
                                                   │
                                                   ▼
                                              data.json (eci)
                                                   │
                          static/script.js: main chart ← data.gaps (unchanged path)
                                             Gap Over Time ← frontier_matches (ECI) / threshold (others)
```

## 7. Failure modes & graceful degradation

| Failure | Handling |
|---|---|
| `eci-public` import fails / not installed | `build_eci_bootstrap` catches `ImportError`, returns `None`. ECI path runs analytical-as-today. |
| `eci_benchmarks.csv` unreachable or schema drift | Caught, returns `None`, analytical fallback. |
| Anchor model/benchmark missing from CSV | `fit_eci_model` raises `ValueError`; caught, returns `None`. |
| Fit numerically fails on some resamples | `capability_samples` shorter than `B`; we use what's returned. If empty, return `None`. |
| Displayed model absent from in-house fit | `prob_exceeds` returns `None` for that pair → predicate falls back to analytical/threshold for that pair only. Logged via match rate. |
| `bootstrap is None` end-to-end | `frontier_matches` omitted from `data.json`; frontend falls back to JS threshold. Output is identical to today. |

The daily digest is therefore strictly **fail-open**: the worst case reproduces
current behavior.

## 8. Testing

- **Regression:** the entire existing `tests/test_gap_calculations.py` must stay
  green unchanged (predicate signature + no-bootstrap path preserved).
- **New unit tests** (`tests/test_eci_bootstrap.py`):
  - `EciBootstrap.prob_exceeds` returns the correct paired fraction and `None`
    for missing names.
  - `_open_caught_up` prefers the bootstrap verdict when draws exist; falls back
    to analytical when `prob_exceeds` is `None`; falls back to threshold when no
    std.
  - `build_eci_bootstrap` returns `None` (no raise) when the fit/fetch is
    monkeypatched to raise.
  - Cache round-trip: build → write → reload by hash skips refit.
- **New integration test:** an ECI-shaped frontier with a hand-built
  `EciBootstrap` whose draws flip a near-boundary pair relative to the 1-point
  threshold, asserting `gaps`/`statistics`/`frontier_matches` reflect the
  bootstrap verdict.
- **Frontend:** extend `tests/test_index_html_structure.py` only if needed; the
  JS change is guarded by `frontier_matches` presence and falls back cleanly.
- **Senior review:** run `gemini -p` on `scripts/eci_bootstrap.py` and the
  `update_data.py` diff per `CLAUDE.md` before commit.
- **CI runtime:** measure the first GitHub Actions run with `B=500`; if it
  exceeds a comfortable budget, reduce `B` (single constant) — no design change.

## 9. Out of scope / accepted trade-offs

- Displayed ECI numbers stay Epoch-published (hybrid by design). A near-boundary
  verdict can therefore come from a fit centered microscopically differently
  from the published point; the chart note makes this explicit.
- No change to any non-ECI benchmark, to the survival-analysis current-gap
  estimator, to trend calculations, or to the China-framing *definition* (only
  its match verdicts shift to bootstrap, consistent with the open/closed view).
- We compare capabilities (scale-invariant), not ECI-scaled draws, so
  `compute_eci_scores` is not on the critical path.

## 10. Files touched

| File | Change |
|---|---|
| `scripts/eci_bootstrap.py` | **New** — `eci-public` wrapper + `EciBootstrap`. |
| `scripts/update_data.py` | Extend `_open_caught_up`; thread `bootstrap` through 4 functions; add `build_frontier_match_map`; wire into `process_data`; emit `frontier_matches`. |
| `static/script.js` | `renderHistoricalChart`: server match map for ECI, JS threshold fallback otherwise; chart note. |
| `requirements.txt` | Pin `eci` from git. |
| `.gitignore` | Ignore `data/eci_bootstrap_cache.npz`. |
| `tests/test_eci_bootstrap.py` | **New** unit + integration tests. |
| `README.md` | Note the ECI bootstrap criterion + attribution to `eci-public`. |
