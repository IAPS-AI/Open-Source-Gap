# Development Protocol

## Review
- No `/code-review` pass is required for any change. Verify with the test suite plus a manual read of the diff.
- Never spawn multi-agent or multi-lens review workflows unless I explicitly ask for one in that message.

## Workflow
1. **Implement** - Write the code, with tests for anything that changes published numbers.
2. **Verify** - Run the test suite (`python -m pytest tests/ -q`) and read the diff.

## Project Notes
- The site is ECI-only: one static page (`index.html`, `static/`) that reads `data.json` and shows two comparisons, open vs closed and China vs U.S. (`#china-us`), both computed by the same method in `scripts/gap.py`. There is no Flask app and no other benchmark.
- `data.json` is served straight from the repo by GitHub Pages and regenerated daily by CI; new pipeline fields appear on the site only after a regeneration (locally: `python scripts/update_data.py`, no secrets needed).
- Pages serves `main` as it is, so bump the `?v=` number on both asset URLs in `index.html` whenever `static/` changes. `tests/test_site.py` guards the element ids, the asset URLs, and that `data.json` matches what the script reads.
- Open/closed flags come only from Epoch's labels in `benchmarked_models.csv` plus `ACCESSIBILITY_OVERRIDES`; there are no org/name heuristics. The daily build fails on purpose (`AccessibilitySourceError`) if those labels go missing. Fix the source; don't add guesses. A new Epoch label shows up as an "Unrecognized accessibility label" warning in the Actions run and needs a decision in `scripts/accessibility.py`.
- Models are counted per `model_group`, the unit the ECI is fitted on and the name the bootstrap draws are keyed by.
- The daily gap is a sawtooth, so the page does not plot it. The lower chart is its 12-month average with a band of one standard deviation (`average.months` and `average.spread` in `data.json`), and the daily value appears only for the selected day.
- The page lists every scored model (`models` in `data.json`) and keys each row to its side of the comparison on screen. Those grouping rules are written twice, in `COMPARISONS` (`scripts/gap.py`) and in each `side` under `WORDS` (`static/script.js`); `tests/test_site.py` checks that they agree.
- China vs U.S. groups models by Epoch's `Country (of organization)` label, open and closed alike. There are no org-name lists; a model with no country is left out of that comparison.
- The China vs U.S. headline is the trendline gap (`trend_lag` in `scripts/gap.py`): straight-line fits to each country's record-setters released since 2024, read at the latest Chinese record. A record counts only if it beats every earlier model from that country, including pre-2024 ones. The open vs closed headline is the 12-month average of the daily gap.
