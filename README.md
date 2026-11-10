# Open Source Gap

How many months open-weight AI models trail the most capable closed models, and Chinese models trail U.S. ones, on the [Epoch Capabilities Index](https://epoch.ai/eci) (ECI). Rebuilt daily from Epoch AI data.

Live at [os-gap.theo-bearman.com](https://os-gap.theo-bearman.com). The China vs U.S. view is at [`#china-us`](https://os-gap.theo-bearman.com/#china-us).

## What it measures

The method is Epoch AI's, from [Open models lag state-of-the-art closed models by 4 months](https://epoch.ai/data-insights/open-closed-eci-gap) (May 29, 2026). It compares a laggard group with a leader group:

1. Each day, take the laggard model with the highest ECI released so far.
2. Find the newest record-setting leader model that it has plausibly caught up to. The gap that day is the time since that leader model's release.
3. A laggard model has caught up when it scores higher than the leader model in at least 5% of 500 paired bootstrap resamples of the benchmark results. The pipeline reruns Epoch's fit with [eci-public](https://github.com/epoch-research/eci-public) to get those resamples; displayed scores stay Epoch's published ones.

For open vs closed, the headline is the daily gap averaged over the 12 months to the latest record-setting release. The page also shows today's gap, the daily series with its 12-month rolling average, and each leader record with the first laggard model to match it.

Two comparisons are published:

| Comparison | Laggards | Leaders | Grouped by |
| --- | --- | --- | --- |
| Open vs closed (Epoch's) | open-weight models | closed models | Epoch's `Model accessibility` label |
| China vs U.S. | models from developers based in China | models from developers based in the U.S. | Epoch's `Country (of organization)` label, with open-weight and closed models counted alike and other countries left out |

The China vs U.S. view leads with a second, smoother measure, the **trendline gap**. A straight line is fitted to each country's record-setting models released since 2024, and the gap is how much earlier the U.S. line reached the level the Chinese line stands at on the date of the latest Chinese record. The page also gives the pace at which that gap is widening or narrowing. A record here means beating every earlier model from that country, including those from before 2024, so the fitted points are the dots on the chart from 2024 on. If either country has fewer than three such records, the view falls back to the 12-month average.

The trendline gap smooths over single releases and is read at the latest Chinese record, so it does not grow while no new Chinese record appears. The daily gap below it moves release by release.

Three properties of the daily series are worth knowing:

- It falls when a laggard model catches up to a newer leader record, and it falls to zero on the day a leader sets a record that the best laggard model had already caught up to.
- Until a laggard model has caught up to any leader record (before April 2024 for open vs closed), the gap counts from the first leader record and is a lower bound. The chart draws that stretch paler, along with the part of the rolling average that includes it.
- The test is statistical, so a new laggard record can fail a comparison its predecessor passed, which moves the gap up.

Models are counted the way the ECI scores them: one entry per `model_group` in Epoch's data, dated by its earliest version. Dated releases of one family, such as GPT-4o (May 2024) and GPT-4o (Nov 2024), are separate entries.

## Layout

| Path | What it is |
| --- | --- |
| `index.html`, `static/` | The page. Plain HTML, CSS, and JavaScript with no dependencies; the charts are hand-drawn SVG. |
| `data.json` | Everything the page shows, written by the pipeline. |
| `scripts/update_data.py` | Entry point: fetch, compute, write `data.json`. |
| `scripts/gap.py` | The gap calculation and the two comparisons. Pure Python. |
| `scripts/eci_data.py` | Fetches and parses Epoch's `benchmarked_models.csv`. |
| `scripts/accessibility.py` | Open or closed, from Epoch's labels only. |
| `scripts/eci_bootstrap.py` | Paired bootstrap draws via eci-public, cached in `data/`. |
| `tests/` | `python -m pytest tests/ -q` |

## Running it

```bash
pip install -r requirements.txt -r requirements-optional.txt pytest
python scripts/update_data.py     # rewrites data.json; no credentials needed
python -m pytest tests/ -q
python -m http.server 8080 --bind 127.0.0.1   # then open http://localhost:8080
```

The first `update_data.py` run refits the ECI 500 times, which takes around 10 minutes. Later runs reuse `data/eci_bootstrap_cache.npz` until Epoch's benchmark data changes.

## Publishing

GitHub Pages serves the `main` branch as it is. `.github/workflows/daily_update.yml` runs the pipeline every day at 00:00 UTC and commits `data.json`.

Browsers cache the script and stylesheet for 10 minutes, so bump the `?v=` number on both asset URLs in `index.html` whenever `static/` changes. `deploy_pages.yml` also publishes on each push and stamps those URLs with the commit, but the next branch build replaces that, so the number in `index.html` is the one that counts.

## Safeguards

- **No guessed labels.** Open or closed comes only from Epoch's `Model accessibility` label, plus a short list of overrides for labels Epoch ships blank. "Limited access" counts as closed. If the label column disappears or fewer than 85% of scored models carry a label, the build fails with `AccessibilitySourceError` and the previous `data.json` stays live. A label the code has not seen is reported as a warning on the Actions run and its models are left out until `scripts/accessibility.py` classifies it.
- **No guessed countries.** Country comes only from Epoch's label. A model without one is left out of the China vs U.S. comparison, and the build fails if the column disappears or more than 15% of models lack it.
- **Visible fallback.** If the bootstrap cannot run, matches use a normal approximation built from the published 90% intervals, which is slightly more permissive. `data.json` records this under `test.paired_bootstrap`, the page says so, and the Actions run carries a warning.
- **All or nothing.** `update_data.py` exits 1 without touching `data.json` if any step fails.

## data.json

Top level: `as_of` (build date, the last day of every series), `days_per_month`, `test` (whether the paired bootstrap ran and with how many resamples), and `comparisons`, which holds `open_closed` and `china_us`. Each comparison has:

| Field | Meaning |
| --- | --- |
| `headline` | Average months behind and average ECI-point gap over the window, with the window's dates and the 5th and 95th percentiles of the daily values. |
| `today` | Gap on `as_of`, with the best laggard model, the leader model it is measured against, and the oldest leader record it has not matched. |
| `segments` | The daily series. Each segment starts on `start` and runs to the next one; within it, months behind on a day is the time since `reference_model` was released, divided by `days_per_month`. |
| `average` | The 12-month rolling average of the daily series: one value per day from `start`, ending on the headline's last day, where it equals the headline. Values before `firm_from` include lower-bound days. |
| `leader_frontier` | Every record-setting leader model, with the first laggard model to catch up, the date, the lag in months, and the test that decided it. |
| `laggard_frontier` | Every record-setting laggard model. |
| `trend` | The trendline gap (`china_us` only; `null` for `open_closed`): `lag_months`, `change_months_per_year` (positive means widening), the date it is read `on`, the `level` of the laggards' line that day, when the leaders' line reached it (`leader_reached`), and for each group the number of fitted `records`, the slope in `eci_per_month`, `r2`, and the two end points of the fitted `line`. |

## Attribution

Data from [Epoch AI](https://epoch.ai), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). If you reuse the figures or the chart, credit Epoch AI as the source.
