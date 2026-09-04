# Development Protocol

## Review Scope (read this first)
Reviews cost a lot of tokens, so they are scoped, not automatic:
- **Review only** changes under `scripts/` that alter published numbers in `data.json` (gap calculations, matching criteria, statistics, fetchers). One pass, one reviewer: run `/code-review` at medium effort on the changed `scripts/` files.
- **Skip review** for UI, copy, CSS, HTML, test-only, and documentation changes. Tests plus a manual read of the diff are enough.
- **One reviewer, not a fan-out.** Never spawn multi-agent or multi-lens review workflows unless I explicitly ask for one in that message.
- Verify each suggestion against the code and tests before implementing it.

## Workflow
1. **Implement** - Write the code, with tests for anything that changes published numbers.
2. **Review** - One `/code-review` pass, only if the change is in scope (see Review Scope).
3. **Iterate** - Implement the suggestions that check out.
4. **Verify** - Run the test suite (`python -m pytest tests/ -q`); do not re-review unless the fix itself changed a calculation.

## When to Trigger a Review
- After changing a gap calculation, matching criterion, or statistic that reaches `data.json`
- After writing or changing a data fetcher or pipeline step
- Not for new charts, page copy, styling, HTML structure, tests, or docs

## Project Notes
- `data.json` is served straight from the repo by GitHub Pages and regenerated daily by CI; new pipeline fields appear on the site only after a regeneration (locally: `python scripts/update_data.py`, no secrets needed).
- Both `index.html` (served by Pages) and `templates/index.html` (Flask) must carry the same chart markup; `tests/test_index_html_structure.py` guards the anchors.
