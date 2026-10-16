# Development Protocol

## Review
- No `/code-review` pass is required for any change. Verify with the test suite plus a manual read of the diff.
- Never spawn multi-agent or multi-lens review workflows unless I explicitly ask for one in that message.

## Workflow
1. **Implement** - Write the code, with tests for anything that changes published numbers.
2. **Verify** - Run the test suite (`python -m pytest tests/ -q`) and read the diff.

## Project Notes
- `data.json` is served straight from the repo by GitHub Pages and regenerated daily by CI; new pipeline fields appear on the site only after a regeneration (locally: `python scripts/update_data.py`, no secrets needed).
- Both `index.html` (served by Pages) and `templates/index.html` (Flask) must carry the same chart markup; `tests/test_index_html_structure.py` guards the anchors.
- Open/closed flags come only from Epoch's labels (`model_metadata.csv` for CSV benchmarks, `benchmarked_models.csv` for ECI) plus `ACCESSIBILITY_OVERRIDES`; there are no org/name heuristics. The daily build fails on purpose (`AccessibilitySourceError`) if those labels go missing. Fix the source; don't add guesses. A new Epoch label shows up as an "Unrecognized accessibility label" warning in the Actions run and needs a decision in `scripts/accessibility.py`.
