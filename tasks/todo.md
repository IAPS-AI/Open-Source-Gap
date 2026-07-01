# Threshold-Crossing Gap Integration (open_closed_gap → this repo)

Spec: `docs/specs/2026-07-01-threshold-gap-integration-design.md`
Plan: `docs/plans/2026-07-01-threshold-gap-integration.md`

- [x] Task 1: Core datapoint computation (`scripts/threshold_gap.py` + tests)
- [x] Task 2: Validity allowlist + pair dedup tests
- [x] Task 3: Gaussian-kernel smoother with bootstrap CI
- [x] Task 4: JSON-ready builders (analysis block + aggregate)
- [x] Task 5: Wire into `scripts/update_data.py`
- [x] Task 6: End-to-end verification against live data
- [x] Task 7: Gemini senior review + fixes
- [x] Task 8: Documentation touch-up

## Review

- New `scripts/threshold_gap.py` (pure calculation) + 33 tests in
  `tests/test_threshold_gap.py`; full suite 102 passed.
- `scripts/update_data.py`: additive `threshold_analysis` per benchmark and
  top-level `threshold_aggregate`, fail-open (never kills the daily pipeline).
  Existing calculations untouched.
- Conflict rule honored: this repo's open/closed classification, live data,
  ×100 percentage scale, month units, and `now()` reference prevail; the
  source's grid/first-crosser/pair-dedup/validity-review/smoother logic ported.
- E2E verification on live data: 238/238 first-crosser dates independently
  recomputed with 0 mismatches; aggregate pools exactly the 5 KEEP benchmarks
  (gpqa_diamond, math_level_5, otis_mock_aime, frontiermath_public,
  metr_time_horizon); medians overall 4.7mo / public 4.3mo / private 7.2mo
  (private > public, same direction as the source's headline finding).
- Gemini review: 1 finding was a misread (dedup already selects the pair's
  highest overall threshold — proven by test), 2 were intentional methodology
  ports, 1 comment-clarity fix applied.
- Frontend chart was shipped 2026-07-01 then removed the same day at user
  request (commit 9449aa4, reverted). The calculation pipeline and data.json
  sections remain; the chart code is recoverable from history if wanted.
- Follow-up (not in scope): external-leaderboard fetchers (WeirdML, ARC-AGI, …)
  if desired.

# Calculation-Logic Audit vs open_closed_gap (2026-07-01)

- [ ] Compare update_data.py / app.py / threshold_gap.py against source analysis.py
- [ ] Write audit report with findings + recommendations
- [ ] Implement clear correctness improvements (if any)
