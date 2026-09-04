# Gap Over Time: gap-by-date timeline and "last time this large" context

**Date:** 2026-09-04
**Status:** Implemented.

## Goal

Make the Gap Over Time chart answer "how far behind the US frontier was the
Chinese frontier on date t, and how has that changed?", and surface datapoints
of the form "the last time the US lead over China was this large was X, N
months ago."

The previous chart plotted one dot per leader-side (US / closed) frontier
release at its release date with y = months until a laggard-side (China /
open) model matched it. That answers "how long did each US model take to be
matched", not "how big was the gap on date t"; recent unmatched US models were
invisible except through the projection wedge.

## Series

The site's headline average gap is the mean of a day-by-day series computed in
`scripts/update_data.py::calculate_gap_metrics` (Epoch AI's method): on each
day, months since the most recent leader frontier (running-max) model that the
day's best laggard model has *plausibly caught up to* (`_open_caught_up`:
paired bootstrap for ECI, CI test when stds exist, else score threshold). Only
its mean and 5th/95th percentiles were published. The new module
`scripts/gap_timeline.py` publishes the series itself.

Computation is event-based: the reference model only changes on frontier
release dates, and between events the gap grows at exactly one month per
month, so the series is a list of `segments` (value on day t =
`(t - reference_date) / DAYS_PER_MONTH`). Segment `end` is exclusive except for
the last segment (`as_of` inclusive). Expanding the segments to daily
resolution reproduces the `calculate_gap_metrics` loop exactly;
`tests/test_gap_timeline.py::test_matches_calculate_gap_metrics_average` pins
it (mean equal to 1e-9 over the same window) and the regenerated `data.json`
shows `context.mean_gap_months == statistics.avg_horizontal_gap_months` for
every benchmark with current data (ECI: 6.0 open-vs-closed, 6.4 China-vs-US).

The window starts on the first day both groups exist and runs to the build
date (`as_of`, computed once per run as `BUILD_AS_OF`), not the last model
date, so the gap keeps accruing after the latest release. `last_model_date` is
published so the frontend can mark the tail after it as data-staleness rather
than measured progress (METR, MATH Level 5 and SWE-bench have stale tails).

Properties of the methodology that the chart labels explicitly:

- the gap drops to zero on the day the leader releases a model the laggard's
  best had already matched (`events[].kind == "leader_release"`);
- it can step *up* when a new laggard record with a higher point estimate fails
  the significance test against a model the previous record had passed
  (`events[].reversal == true`; the CI test is more permissive for larger
  std, so a sharper new model can fail where a noisier old one passed).

## JSON block `gap_timeline` (per benchmark, and per `china_framing`)

```
method, days_per_month, as_of, start, last_model_date, n_days,
segments[]: start, end, gap_start, gap_end, laggard_model/date/score,
            reference_model/date/score, reference_matched
events[]:   date, kind (laggard_release | leader_release), model,
            gap_before, gap_after, reversal, laggard_model, reference_model
leader_frontier[], laggard_frontier[]: model, date, score (running-max records)
context: current_gap_months, current_since, current_laggard_*, current_reference_*,
         current_reference_matched,
         last_at_least {date, gap_months, months_ago, run_since},
         last_at_most  {date, gap_months, months_ago, run_since},
         is_record_high, is_record_low, peak{}, trough{},
         mean_gap_months, median_gap_months, percentile_of_current,
         year_ago{}, trailing_12m{mean, prior_mean, change}
```

`last_at_least` (symmetrically `last_at_most` with <=): let cur = gap(as_of).
Walk back from the day before as_of while gap >= cur; that contiguous run, if
any, is the current >=cur episode and its first day is `run_since`. Then skip
the days with gap < cur; the most recent day with gap >= cur is `date`, with
`gap_months` and `months_ago`. Ties count. On a rising tooth `run_since` is
null and `date` is the most recent earlier day at or above today's level (the
user's phrasing). If the gap dropped today, `run_since` is the start of the
run that just ended. Record semantics come from `is_record_high` /
`is_record_low` (cur vs all earlier days), never from a null `date`.

China framing uses `df_china_us_frontier` (China vs US only), the same
universe as the headline China statistics. Benchmarks that ship no
`china_framing` (all except ECI and METR) get no China timeline; the frontend
then disables the timeline view rather than showing open-vs-closed numbers
under China labels.

## Frontend

- `renderHistoricalChart(data)` dispatches between `renderGapTimelineChart`
  (default, "Gap by date") and the previous chart, renamed
  `renderHistoricalMatchesChart` ("Time to match each release"), via
  `appState.historicalView` and the `#historical-view-toggle` buttons. Shared
  helpers: `drawGapChartAxes`, `drawReleaseTicks` (every release gets a tick;
  only labels are culled), `placeLabelBand`, `drawLabelBand`,
  `setHistoricalMethodNote`.
- Timeline marks: area + sawtooth polyline; drop-event dots with the 2-row
  label band (`Kimi K3`, `5.3 -> 4.4 mo`); hollow laggard-accent ring for
  leader releases already matched; hollow ink ring for reversals; leader
  frontier releases as rotated ticks; a today block in the right margin
  (`6.0 mo`, `matched lead today`, `last >= this: May 2026`); a lookback ring
  on the line at `last_at_least.date` joined to today by a dotted guide at
  today's level; in Current-Gap mode the survival-analysis star labelled
  `Est. 7.8 mo / incl. unmatched (survival)`; an in-SVG caption and `<title>`
  so the PNG export carries the numbers; a crosshair tooltip snapped to the
  day showing the best laggard model and the reference it had caught up to.
- Context strip (`#gap-context`): lead sentence naming what the number
  measures, the survival estimate in Current-Gap mode, and "The last time it
  was this large was April 2026, 4.5 months ago" (record-high / just-dropped /
  never-before variants keyed on the flags); a muted "last this small" line
  only when the lead is below its median; tiles for current lead, last this
  large, peak, low, trailing-12-month average vs the prior 12 months, and
  median with the current percentile.
- Legend row (`#historical-legend`) and a method note under the chart that
  states the definition, that the headline average is the mean of the line,
  and that the line counts only matched leader models so it can sit below the
  survival estimate.

## Tests

`tests/test_gap_timeline.py` (23 tests): degenerate inputs, single segment,
sawtooth geometry and events, lookback semantics including ties, dropped-today,
zero-gap record low, reversal, same-date leader pair, equal-score laggard,
trailing-12-month means, last_model_date, frontier lists, exact agreement with
`calculate_gap_metrics`, JSON serialisability and rounding.
`tests/test_index_html_structure.py` guards the new anchors in both HTML files
and the JS entry points.

## Addendum (same day): accounting for unmatched leader models

The matched lead cannot rise when the leader releases a model the laggard has
not matched (GPT-6 Astra on Sep 3, 2026 left it at 6.0 months). Three
additions make unmatched models visible:

1. **Measured range (band).** Each segment now carries `next_leader_model /
   next_leader_date / next_leader_score` (the earliest leader record released
   after the reference, i.e. the oldest unmatched one) and `lower_start /
   lower_end` (months since it; 0 when the laggard has matched the newest
   leader model). Segments split when that model changes too, so the lower
   series is exact. The chart shades the band between the lower bound and the
   matched lead; the sentence reads "the US lead over China is between 4.4 and
   6.0 months ... has caught up to GPT-5.4 Pro but not GPT-5.5 Pro".
   `context.current_lower_months`, `current_next_leader_*`, `mean_lower_months`.
2. **Expected lead (forecast line).** `gap_timeline.expected_lead.points` is
   the survival-analysis current-gap estimate (`estimate_current_gap`, same
   flags as `calculate_statistics` for the benchmark) recomputed as of every
   7th day using only models released by then; `calculate_horizontal_gaps`
   gained an `as_of` parameter for the age of unmatched models. Drawn as a
   dashed line ending at today's survival estimate (7.8 months), labelled a
   forecast. Its endpoint equals `statistics.current_gap_estimate`.
3. **Score gap view.** `score_gap_timeline` (per benchmark and framing): best
   leader score minus best laggard score on each day, signed, as constant
   `steps` with `events` on every frontier release, the same frontier lists,
   and the same context block with `gap_points` / `change_points` keys
   (`_daily_context` is shared). Its mean equals
   `statistics.avg_vertical_gap` when no day is negative (tested), and its
   current value equals `current_vertical_gap`. The view is the second toggle
   option ("Score gap"); the lead sentence answers "how big is the score gap
   now and when was the last time it was this large": 11.6 ECI points on Sep
   4, 2026, last at least this large in December 2024.

The toggle now has three options: Months behind (default), Score gap, Time
to match each release. Units for the score view come from the benchmark
metadata (ECI points; percentage points for 0-100 benchmarks; otherwise the
unit string).

## Follow-ups (not done)

- A strict-criterion band or a smoothed series alongside the headline series.
- China framing for the seven benchmarks that ship none (the page already shows
  open-vs-closed numbers under China labels for those; pre-existing).
- The survival prior still uses the per-release matching that skips laggards
  released on or before the leader date (`calculate_horizontal_gaps`).
