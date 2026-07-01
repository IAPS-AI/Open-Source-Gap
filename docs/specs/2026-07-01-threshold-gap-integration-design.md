# Threshold-Crossing Gap Analysis Integration — Design

**Date:** 2026-07-01
**Source:** https://github.com/htihle/open_closed_gap (Håvard Ihle's open-vs-closed gap analysis)
**Rule:** where calculation logic differs between the two projects, **this repo's logic prevails**.

## Goal

Integrate the *calculation logic* of open_closed_gap — the "backward-facing delay"
threshold-crossing methodology — into this repo's live data pipeline, so its outputs
land in `data.json` alongside the existing per-model matching analysis. No existing
calculation changes.

## The two methodologies

| | This repo (existing, unchanged) | open_closed_gap (being integrated) |
|---|---|---|
| Question | For each closed frontier model, when did an open model *statistically catch up*? | For each score threshold T, how much later did the first open model reach T than the first closed model? |
| Matching | ECI bootstrap ≥5% rule / analytical CI / point threshold | Hard `score >= T`, earliest release date wins |
| Universe | Frontier (running-max) models | All models (equivalent: a first-crosser is always a within-group running-max record) |
| Trend | Day-by-day gap averaging, linear/exp regression | Gaussian-kernel smoothing (σ=60d) + bootstrap CI over (benchmark × threshold) datapoints |

These answer different questions, so the integration is **additive**: new module, new
`data.json` keys, existing outputs byte-identical in structure.

## What is ported from open_closed_gap

From `open_vs_closed/analysis.py`:

1. **`compute_datapoints` core** — per benchmark:
   - Fixed score-threshold grid (`auto_thresholds`: 0.05 steps on [0,1]-scaled data
     → ×100 here since this repo scales to percentages; 5.0 steps on 0–110 data;
     else 21-point linspace interior). Per-benchmark hand-set grids where the source
     had them (METR: [2,5,10,20,50,100,200,500] minutes).
   - `first_crosser`: earliest-dated model in each group with `score >= T`.
   - `gap_days = first_open_date - first_closed_date` (negative allowed — open led).
   - **Still-open thresholds**: closed crossed T, no open model has → recorded with
     `still_open: true` and the ongoing gap measured against `as_of`.
   - **Validity**: explicit `accepted_thresholds` allowlist (from the source repo's
     per-pair manual review) overrides `validity_floor`; fallback floor is the 10th
     percentile of observed scores (`auto_validity_floor`).
   - **Pair dedup**: a (closed, open) pair recurs across a contiguous run of
     thresholds with an identical gap; keep ONE representative datapoint per accepted
     pair at the HIGHEST threshold the pair spans (even if that specific threshold is
     not itself in the allowlist — matches source semantics of "expand accepted anchor
     pairs to the pair's full threshold span"). All other rows are `accepted: false`.
2. **`gaussian_smooth_with_ci`** — Nadaraya-Watson Gaussian-kernel smoother with
   bootstrap CI, using the source's published-figure parameters: bandwidth 60 days,
   7-day grid, n_boot=5000, 90% CI, min_ess=2.0, fixed seed 0. NaN outside data
   support (no extrapolation).
3. **Review verdicts** (KEEP/DISCARD status + accepted thresholds + public/private
   data-access labels) mapped to this repo's benchmark ids, thresholds rescaled ×100
   where this repo stores percentages:

   | this repo id | source title | status | accepted thresholds (this repo's scale) | access |
   |---|---|---|---|---|
   | `gpqa_diamond` | GPQA Diamond | KEEP | 35…90 (step 5) | public |
   | `math_level_5` | MATH Level 5 | KEEP | 15,20,…,65,80,90,95 | public |
   | `otis_mock_aime` | OTIS Mock AIME 2024-2025 | KEEP | 5,10,15,20,45,…,95 | public |
   | `frontiermath_public` | FrontierMath | KEEP | 10,15,25,30,35 | private |
   | `metr_time_horizon` | METR Time Horizons | KEEP | 2,5,10,20,50 (minutes, no rescale) | private |
   | `swe_bench_verified` | SWE-Bench Verified | DISCARD | — (auto floor) | public |
   | `simpleqa_verified` | SimpleQA Verified | DISCARD (review accepted 0) | — (floor 20) | public |
   | `chess_puzzles` | Chess Puzzles (Epoch) | DISCARD | — (auto floor) | private |
   | `eci` | — (not in source) | EXCLUDED from aggregate (composite index; would double-count constituent benchmarks) | — (auto grid + floor) | — |

   Validity floors from the source, rescaled: gpqa 20, math 10, otis 5,
   frontiermath 10, simpleqa 20, metr 1.5.

   DISCARD/EXCLUDED benchmarks still get full per-benchmark datapoints in
   `data.json` — they are only excluded from the cross-benchmark aggregate,
   mirroring the source's KEEP/DISCARD handling.

## Where this repo's logic prevails (conflicts resolved)

- **Open/closed classification:** this repo's existing `Open`/`is_open` columns
  (contains-"Open" match; blank accessibility → "Open weights (assumed)" unless from
  a known closed lab; `_is_open_model` heuristics for CSV benchmarks). The source's
  `startswith("Open weights")` + ECI-index join + `data_patches/` hand overrides are
  NOT ported. The threshold module takes the already-classified dataframe as input.
- **Data source:** live daily fetch (benchmarked_models.csv, benchmark_data.zip,
  METR). The source's frozen CSV snapshot, external leaderboards not already in this
  repo, and data patches are NOT imported.
- **Units and output format:** `gap_months` (via this repo's
  `DAYS_PER_MONTH = 365.25/12`) as the primary field, `gap_days` retained; ISO date
  strings; JSON-safe values (no NaN/NaT — `null` instead).
- **"Now" reference:** still-open gaps measured against runtime `now()` (or an
  injected `as_of` for tests), not the source's frozen `AS_OF` snapshot date.
- **Score scale:** this repo's ×100 percentage scale for 0–1 benchmarks; all source
  thresholds/floors rescaled accordingly.
- **Existing calculations:** `calculate_horizontal_gaps`, `_open_caught_up`,
  bootstrap criterion, survival analysis, historical gaps, trends, China framing —
  all unchanged.

## Architecture

New module **`scripts/threshold_gap.py`** (pure functions, numpy/pandas only, no I/O):

- `auto_thresholds(scores) -> np.ndarray`
- `auto_validity_floor(scores) -> float`
- `first_crosser(df, threshold, score_col, date_col) -> row | None`
- `compute_threshold_datapoints(df, *, score_col, model_col, date_col, thresholds=None, accepted_thresholds=None, validity_floor=None, as_of=None) -> list[dict]`
  — one dict per threshold: `threshold, first_closed_model, first_closed_date,
  first_open_model, first_open_date, gap_days, gap_months, valid, accepted,
  still_open, validity_floor`. Expects a dataframe with an `Open` boolean column
  (this repo's classification, already applied).
- `gaussian_smooth_with_ci(dates, values, *, bandwidth_days=60, step_days=7, n_boot=5000, ci=0.90, min_ess=2.0, seed=0) -> (grid, mean, lo, hi)`
- `summarize_datapoints(datapoints) -> dict` — `n_accepted, n_still_open,
  median_gap_months` (over accepted).
- `THRESHOLD_REVIEW: dict[benchmark_id, dict]` — the mapping table above
  (`accepted_thresholds`, `validity_floor`, `thresholds`, `keep`, `data_access`).
- `build_threshold_analysis(df, benchmark_id, *, score_col, model_col, as_of=None) -> dict`
  — convenience wrapper: looks up `THRESHOLD_REVIEW`, computes datapoints + summary,
  returns the JSON-ready `threshold_analysis` block.
- `build_threshold_aggregate(per_benchmark: dict[str, dict], as_of=None) -> dict`
  — pools accepted datapoints from `keep` benchmarks, computes overall +
  public/private smoothed trend series and medians.

**`scripts/update_data.py` changes** (small, additive):

- `process_data` (ECI), `process_benchmark_data`, `process_metr_data`: after building
  `df_combined`, attach `benchmark["threshold_analysis"] = build_threshold_analysis(...)`.
- `process_all_benchmarks`: after all benchmarks processed, attach top-level
  `data["threshold_aggregate"] = build_threshold_aggregate(...)`.
- Failure isolation: threshold analysis wrapped so an error degrades to
  `threshold_analysis: null` + logged warning, never kills the daily pipeline.

**`data.json` additions** (all additive; frontend ignores unknown keys):

```jsonc
"benchmarks": {
  "<id>": {
    // ...existing keys unchanged...
    "threshold_analysis": {
      "config": {"keep": true, "data_access": "public",
                  "validity_floor": 20.0,
                  "accepted_thresholds": [35.0, 40.0, ...],  // null when floor-based
                  "source_review": "open_closed_gap"},
      "datapoints": [
        {"threshold": 35.0, "first_closed_model": "...",
         "first_closed_date": "2023-...", "first_open_model": "...",
         "first_open_date": "2024-...", "gap_days": 312, "gap_months": 10.3,
         "valid": true, "accepted": true, "still_open": false}
      ],
      "summary": {"n_accepted": 9, "n_still_open": 2, "median_gap_months": 8.9}
    }
  }
},
"threshold_aggregate": {
  "datapoints": [ {"benchmark_id": "...", "benchmark_name": "...",
                    "data_access": "public", "threshold": 35.0,
                    "first_open_date": "...", "gap_months": 10.3, /*models+dates*/ } ],
  "trends": {
    "overall":  {"points": [{"date": "...", "mean": 9.1, "lo": 7.2, "hi": 11.0}], "n_datapoints": 41, "n_benchmarks": 5},
    "public":   { /* same shape */ },
    "private":  { /* same shape */ }
  },
  "medians": {"overall": 9.0, "public": 7.1, "private": 13.2},
  "parameters": {"bandwidth_days": 60, "step_days": 7, "n_boot": 5000,
                  "ci": 0.9, "min_ess": 2.0, "seed": 0}
}
```

## Error handling

- Empty group (no open or no closed models): return `[]` datapoints.
- No closed crosser at T: threshold skipped (source behavior).
- Trend sides with no datapoints: key omitted from `trends`.
- All values JSON-safe: `None` instead of NaN/NaT; dates ISO strings.
- Pipeline never fails due to threshold analysis (try/except + warning).

## Testing

`tests/test_threshold_gap.py`, same style as `test_gap_calculations.py`:

- first-crosser selection (earliest date wins, not highest score)
- gap sign (negative when open led)
- still-open rows (closed crossed, open pending; gap vs injected `as_of`)
- skip thresholds no closed model reached
- allowlist validity vs floor validity; float-rounding-safe membership
- pair dedup: representative at pair's highest threshold; duplicates not accepted;
  dedup spans allowlist gaps (source's "expand to pair's span" semantics)
- auto thresholds for 0–1(→×100) and minute-scale data
- smoother: constant series → constant mean; NaN outside min_ess support;
  deterministic under fixed seed; CI brackets mean
- aggregate: keep-filtering (DISCARD + eci excluded), public/private split, medians
- JSON-safety of the full built blocks (`json.dumps` round-trip)

Verification: run full test suite; run `update_data.py` end-to-end and inspect the
new `data.json` sections.

## Out of scope (follow-ups)

- Frontend chart for the threshold analysis (data is published in `data.json`).
- The source repo's matplotlib figures, icons, milestones annotations.
- Its open-*Chinese*-vs-closed variant (this repo already has a China framing with
  its own methodology, which prevails).
- External-leaderboard benchmarks this repo doesn't already fetch (WeirdML,
  ARC-AGI, Aider Polyglot, …) — the module supports them if fetchers are added later.
