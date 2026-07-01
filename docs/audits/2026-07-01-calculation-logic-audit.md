# Calculation-Logic Audit vs open_closed_gap

**Date:** 2026-07-01
**Reference:** https://github.com/htihle/open_closed_gap @ `ae8586d`
**Scope:** every calculation path in this repo (`scripts/update_data.py`,
`scripts/csv_benchmark_fetcher.py`, `scripts/metr_fetcher.py`,
`scripts/threshold_gap.py`, `app.py`) compared against the source repo's
methodology, with empirical cross-checks against its hand-curated data.

## Method

- Line-level comparison of each calculation function against the source's
  `open_vs_closed/analysis.py`, `_expand_accepted.py`, `regenerate_data.py`.
- Empirical classification cross-check: every model row in our seven Epoch
  benchmark CSVs and in `benchmarked_models.csv` classified by our code and
  compared against the source's curated `epoch_capabilities_index.csv`
  (frozen 2026-05-22 **with** its 37 hand-verified accessibility overrides).
- Impact scan of the currently published `data.json`.

## Findings

### F1 — CRITICAL: benchmark open/closed classification is heuristic and wrong

Epoch's per-benchmark CSVs (gpqa_diamond.csv, math_level_5.csv, …) ship **no
`Model accessibility` column**, so `csv_benchmark_fetcher._is_open_model`
always falls through to name/org heuristics. Those heuristics are wrong for
every mixed-release lab:

| Misclassification | Models (examples) | Cause |
|---|---|---|
| API-only → "open" | qwen-turbo/plus/max, qwen3-max, qwen3.5/3.6 plus/flash/max-preview, qwq-plus, mistral-large-2402, mistral-medium-2505, ministral-3b, muse-spark | `'qwen'/'alibaba'/'mistral'` in OPEN_SOURCE_ORGS |
| Open-weights → "closed" | openai/gpt-oss-120b, Phi-3-medium, phi-4, WizardLM-2-8x22B | `'openai'/'microsoft'` in CLOSED_SOURCE_ORGS |

Disagreements vs the curated index: GPQA 18, OTIS 14, MATH L5 10,
FrontierMath 9, SimpleQA 8, Chess 7, SWE-Bench 2.

**Impact on published numbers:** 11 matched gaps in the live `data.json`
use an API-only model as the "open" matcher (e.g. GPQA: gpt-4-0314 "matched"
by mistral-large-2402 at 11.5 mo; MATH L5: o4-mini "matched" by qwen3-max at
5.3 mo; FrontierMath: gpt-5.2 "matched" by muse-spark at 3.9 mo). All of these
make gaps look **shorter** than reality. Misclassified frontier models also
appear on the charts with the wrong colour, and WizardLM-2-8x22B — the source
repo's own MATH L5 open first-crosser — sits on our *closed* side.

**Source repo's approach (adopted):** join each benchmark CSV to
`epoch_capabilities_index.csv` (present in the same `benchmark_data.zip` we
already download) and classify from `Model accessibility`; models it cannot
classify are **excluded**, not guessed.

**Fix implemented:** classification precedence per model row is now
(1) the CSV's own accessibility column if ever present → (2) the live Epoch
capabilities index → (3) a vendored override map (the source repo's 37
hand-verified rows plus this repo's additions for models Epoch leaves blank)
→ (4) narrowed heuristics (unambiguous name patterns/orgs only; `qwen`,
`alibaba`, `mistral`, `moonshot`, `zhipu` removed from the open list,
`microsoft` removed from the closed list) → (5) otherwise **excluded** with a
logged count. Shared module: `scripts/accessibility.py`.

### F2 — HIGH (latent): ECI "blank accessibility → assumed open" rule

`benchmarked_models.csv` has 8 blank-accessibility rows today; the rule in
`fetch_eci_data` marked 6 of them "Open weights (assumed)". Four are Alibaba's
**hosted/API-only** models (Qwen 3.5 Plus "(hosted 397B-A17B)", Qwen 3.5/3.6
Flash, Qwen 3.6 Max Preview) — the curated index marks their slugs API access.
The ECI headline is *currently* unaffected only because none of them beats the
genuine open frontier (Kimi K2.x / GLM-5.x / DeepSeek), but one high-ECI hosted
release would silently flip the headline gap.

`'Unreleased'` models (e.g. PaLM 62B per the source's override) were also
being classified **closed**; the source excludes them.

**Fix implemented:** blanks are filled from the vendored override map (all 8
current blanks covered, including `o3-pro` and `Gemini 3.5 Flash` → API
access); any model still unclassifiable (blank/`Unreleased`/unknown value) is
**excluded** from the analysis with a logged name list. The
"assumed open" rule and the `KNOWN_CLOSED_LABS` special case are removed.

### F3 — MEDIUM: `app.py` `/api/data` computes with drifted legacy logic

`app.py` carries an old copy of the pipeline that has diverged from
`scripts/update_data.py`:

- blank accessibility → **closed** (opposite of the pipeline rule);
- `calculate_horizontal_gaps` uses only the raw 1-point tolerance — no
  bootstrap, no analytical-CI criterion (`match_type` is always `'exact'`);
- `calculate_trends` lacks the dynamic midpoint split and returns
  `float('inf')` doubling times (invalid strict JSON);
- fetch lacks the browser User-Agent workaround and the all_ai_models backfill.

The frontend never calls `/api/data` (it fetches the static `data.json`), so
this is a latent inconsistency, not a live bug. **Recommendation (not
implemented — endpoint removal is an owner decision):** delete `/api/data`, or
make it serve the prebuilt `data.json` like the `/data.json` route.

### F4 — LOW: METR fetcher name-pattern classification

`metr_fetcher.OPEN_MODEL_PATTERNS` includes `mistral_large` and bare `qwen` —
the same mixed-lab hazard as F1 if METR ever adds hosted variants. METR's
model set is small and currently classified correctly. Recommendation: review
if METR adds Alibaba/Mistral hosted models. Not changed.

### F5 — Methodology deltas (intentional, no action)

- `calculate_historical_gaps` clamps negative gaps to 0; the source keeps
  negative gaps (open led). Ours is a deliberate choice for the
  "how far behind" timeline; the threshold analysis keeps negatives, matching
  the source.
- The threshold-crossing port (`threshold_gap.py`) was re-verified against the
  source HEAD (`ae8586d`): accepted-threshold lists, `_expand_accepted`
  anchor→span semantics, grid/floor logic, pair dedup, and smoother parameters
  (σ=60d, 7d grid, 5000 boots, 90% CI, min_ess 2, seed 0) all match. The
  source's newer HLE/BBH/WebDev changes concern benchmarks this repo does not
  ingest.
- `DAYS_PER_MONTH = 365.25/12` identical in both projects.
- The source's `data_patches/` extra rows (HLE, WeirdML) are dataset curation
  for benchmarks we don't ingest; its accessibility overrides ARE adopted
  (F1/F2).
- `get_rank` is O(n²) in Python; fine at current sizes (~700 rows max).

## Verification of fixes

- 20 new unit tests for `classify_accessibility`, override coverage, and the
  fetcher classification chain (`tests/test_accessibility.py`); full suite
  122 passed.
- Full pipeline re-run against live Epoch data:
  - residual misclassifications (wrong-open as matcher/frontier, wrong-closed
    open models): **0** (was 11 wrong matched gaps + misclassified frontier
    models across 7 benchmarks);
  - corrected first-crossers now agree with the source repo's independently
    curated pairs — e.g. GPQA T=35: gpt-4-0314 → **WizardLM-2-8x22B** 13.1 mo
    (previously mistral-large-2402, 11.5 mo); gpt-4-0314's per-model matcher is
    now dbrx-instruct (12.5 mo);
  - ECI exclusions: `Nemotron-4 15B` and `Llama 2-34B` — both announced but
    never released, previously counted on the closed side;
  - threshold aggregate: 46 datapoints, medians overall 4.6 / public 4.3 /
    private 7.2 months (stable).

## Data provenance note

The vendored override rows originate from the source repo's
`data_patches/eci_accessibility_overrides.csv` (hand-verified against public
release information by its author) plus this repo's additions for
`benchmarked_models.csv` display names. Factual accessibility labels; see
`scripts/accessibility.py` for the list and provenance comments.
