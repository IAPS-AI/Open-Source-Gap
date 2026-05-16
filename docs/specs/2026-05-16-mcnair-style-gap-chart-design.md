# Gap-Over-Time chart redesign — McNair-style replication

**Date:** 2026-05-16
**Status:** Design approved; implementing.

## Goal

Replace the current Plotly "Gap Over Time" chart (`#historical-chart`) with a
custom inline-SVG chart visually matching the McNair Center's China page chart
(<https://mcnair.center/china/>): black gap line over a soft-green area fill,
green dots at each laggard-side frontier release labeled above the line with a
2-row collision-aware label band, rotated leader-side model ticks below the
plot, Tufte-minimal axis styling, and a rich hover tooltip.

The chart must continue to respect the existing Framing toggle
(Open vs Closed / China vs US) and Gap Metric toggle (Average / Current Gap
Est.).

## Non-goals

- No changes to `#chart`, `#trend-chart`, or `#distribution-chart`.
- No server-side / Python data-pipeline changes. All transforms client-side.
- No new dependencies.

## Architecture

- Replace the body of `renderHistoricalChart(data)` in `static/script.js`.
  The function signature, call sites, and target div (`#historical-chart`)
  stay the same.
- Implementation is inline SVG via `document.createElementNS`. Plotly is no
  longer used for this chart. Plotly remains in use elsewhere.
- All code lives in `static/script.js` alongside the other render functions.
  Helpers (`monthShort`, `quarterTicks`, `shortenModelName`, etc.) live as
  local functions in the same module.
- A small `<style>` block in `static/style.css` adds the chart-specific
  classes (`.gap-area`, `.gap-line`, `.cn-dot`, `.us-tick`, `.label-cn`,
  `.label-cn-gap`, `.label-us`, `.label-us-box`, `.label-stub`,
  `.hover-line`, `.hover-target`, `.grid-line`, `.axis-label`,
  `.axis-title`). Color tokens live in `:root` (`--c-gap`, `--c-us`,
  `--c-cn`, `--c-cn-soft`).

## Data derivation (client-side, from `data.benchmarks[id].models`)

The API already returns the array of models with `eci`, `date`, `is_open`,
`is_china`, and the gap series in `historical_gaps`. We derive two additional
arrays in the render function:

- **`laggardFrontier[]`**: walk all models with `is_open === true`
  (Open-vs-Closed framing) or `is_china === true` (China-vs-US framing),
  sorted by date. Keep only models that raise the running max score within
  the laggard group. For each, attach `gap_months` and `prev_gap_months` by
  matching to the nearest entry in `historical_gaps`.
- **`leaderFrontier[]`**: same as above for the complement group
  (closed / non-China / US). Used for the bottom rotated ticks.

The gap-line series is the existing `historical_gaps` payload — no
recomputation.

For tooltips, for each laggard frontier release we find the first leader
model whose score ≥ laggard score. That's the model the release "caught up
to." Computed lazily and memoized on first hover.

## Visual specification

Geometry (matches McNair):

- Outer viewBox 1200 × 630, `preserveAspectRatio="xMidYMid meet"`, scales to
  parent width.
- Margins: top 76 (2-row laggard label band), right 90, bottom 140 (rotated
  leader labels + x-axis title), left 84 (rotated y-axis title + numeric
  ticks).
- Y axis: 0 to ceil(max gap / 3) × 3 + 1, ticks every 3 months. Rangemode
  always `tozero`.
- X axis: from earliest laggard release on or after `PARITY_START`
  (2024-04-01 for ECI; for other benchmarks, the first laggard frontier
  date) to the latest laggard release. Quarterly gridlines/ticks; January
  labeled as year (bold), other quarters as month abbreviation.

Elements:

- `polygon.gap-area` — area between gap line and y=0, fill
  `rgba(138, 206, 0, 0.22)`.
- `polyline.gap-line` — gap line, stroke `var(--ink)`, width 1.75.
- `line.us-tick` — short vertical mark at the bottom for each leader
  frontier release.
- `text.label-us` — leader model name, rotated 40° clockwise. White mask
  rect (`rect.label-us-box`) drawn behind to prevent gridline/label
  collision. Greedy left-to-right culling: if a label would overlap a
  previously placed one, drop it and its tick together.
- `circle.cn-dot` — green dot per laggard frontier release.
- `text.label-cn` + `text.label-cn-gap` — name + "prev → curr mo"
  (single value for the first release) in a 2-row band above the plot.
  Greedy assignment: prefer lower row, fall back to upper row, drop only
  if neither fits.
- `line.label-stub` — thin leader from label down toward the dot.
- `line.hover-line` — dashed vertical line on hover, opacity 0 when idle.
- `rect.hover-target` — transparent hit-rect over the plot area.

Tooltip is an absolutely-positioned `<div>` inside
`#historical-chart-container`, hidden by default, populated on mousemove
with the snapped point's data:

```
<gap_months> months behind <leader> frontier
at <laggard release> release
[laggard swatch] LAGGARD RELEASE
   <model name>
   <date> · ECI <score>
[leader swatch] FIRST <LEADER> MODEL TO REACH THIS ECI
   <model name>
   <date> · ECI <score>
```

## Framing and metric toggles

- `appState.framing === 'open'`: laggard = open, leader = closed. Y-axis
  title: "Months open-source lagged closed frontier at each open release."
  Tooltip header: "<gap> months behind closed frontier at <model> release".
- `appState.framing === 'china'`: laggard = China, leader = US. Y-axis
  title: "Months China lagged the US frontier at each Chinese release."
- `appState.gapMetric === 'average'`: chart ends at the last laggard
  release; the existing average reference line is dropped (replaced by the
  visual story of the line+dots).
- `appState.gapMetric === 'current'`: the line extends from the last
  laggard release to today as a soft dashed segment. A star marker is
  placed at (today, `current_gap_estimate.estimated_current_gap`). A soft
  red wedge between the last release and today shows the uncertainty range
  (`min_current_gap` to `estimated_current_gap`).

Toggle handlers already call `renderHistoricalChart(currentData)`; no
changes needed to those handlers.

## PNG export

`<button class="chart-download">↓ Download PNG</button>` placed in a
`.card-foot` row directly below the SVG.

On click:

1. Clone the SVG, inline computed styles for the classes we use, set
   `width`/`height` attributes for 2× resolution (2400 × 1260).
2. Serialize with `XMLSerializer`, wrap in
   `data:image/svg+xml;charset=utf-8,...`.
3. Load into an `Image`, draw to a `<canvas>`, then
   `canvas.toBlob('image/png')` → `URL.createObjectURL(blob)` →
   `<a download="gap-over-time-<framing>-<YYYY-MM-DD>.png">`.click().
4. Revoke the object URL after a short delay.

## HTML changes (`templates/index.html`)

Replace the existing `#historical-chart-container` markup with:

```html
<div class="header">
  <h1>Gap Over Time</h1>
  <p class="subtitle">How the performance gap has evolved historically.</p>
</div>
<div class="card chart-card" id="historical-chart-card">
  <div id="historical-chart-container">
    <div id="historical-chart"></div>
    <div id="historical-tooltip" class="tooltip" hidden></div>
  </div>
  <div class="card-foot">
    <button type="button" class="chart-download" id="historical-chart-download" aria-label="Download high-resolution PNG">
      <span class="chart-download__arrow" aria-hidden="true">↓</span>
      <span class="chart-download__label">Download PNG</span>
    </button>
  </div>
</div>
```

The wrapper `.card.chart-card` mirrors the McNair structure (header inside a
card with a footer toolbar). Existing `.header` styling outside the card
keeps the section title consistent with the rest of the page.

## CSS additions (`static/style.css`)

Adds, at the bottom of the file, the chart-specific class definitions
listed in *Architecture* above plus a small `.tooltip` definition mirroring
McNair's (absolute positioning, paper background, faint shadow,
tabular-nums). Reuses existing CSS variables (`--ink`, `--paper`, etc.)
where they exist; falls back to literal colors otherwise.

## Edge cases

- `historical_gaps` empty → keep existing "No historical data available"
  fallback (render a `<p>` into `#historical-chart`).
- Laggard frontier empty (single-laggard benchmark) → render line+area only,
  no dots/labels.
- Leader frontier empty → render the chart without bottom ticks; do not
  crash on `matchedLeaderFor`.
- Two laggard releases on the same date → keep both; greedy label placement
  drops the lower-priority label if it can't fit.
- `current_gap_estimate` missing in current mode → behave like average mode
  but skip the star marker.

## Testing

The existing test harness is Python-only (`tests/test_gap_calculations.py`).
No JS test infrastructure exists; introducing one for this change is out of
scope.

Instead:

- Add a Python test `tests/test_index_html_structure.py` that loads the
  served `index.html` (via Flask test client) and asserts:
  - `#historical-chart-card` is present.
  - `#historical-chart-download` is present.
  - `script.js` is referenced.
- Manual smoke pass in `app.py`'s dev server: load each framing/metric
  combination, hover the chart, click download, eyeball the PNG.

## Open questions

None. Design approved (Sections 1–7).
