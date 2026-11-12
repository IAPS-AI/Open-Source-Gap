"""Rebuild data.json: fetch Epoch's ECI scores, measure the open-closed gap,
write the result for the site.

    python scripts/update_data.py

Needs no credentials. Exits 1 without touching data.json if anything fails,
so a broken source never replaces good data. The daily workflow commits
data.json only when this succeeds.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from eci_bootstrap import build_eci_bootstrap
from eci_data import BENCHMARKED_MODELS_URL, ECI_BENCHMARKS_URL, decode, fetch, load_models
from gap import build_payload

logger = logging.getLogger("update_data")

ROOT = Path(__file__).resolve().parent.parent
DATA_JSON = ROOT / "data.json"
CACHE_DIR = ROOT / "data"


def load_draws(cache_dir: Optional[Path]):
    """Paired bootstrap draws, or None (with a visible warning) when they
    cannot be built. The gap is still computed, from the analytical test."""
    try:
        draws = build_eci_bootstrap(fetch(ECI_BENCHMARKS_URL, timeout=120), cache_dir=cache_dir)
    except Exception as error:
        logger.warning("Could not fetch eci_benchmarks.csv: %s", error)
        draws = None
    if draws is None:
        message = ("paired bootstrap unavailable; matches fall back to the analytical "
                   "test on published intervals, which is slightly more permissive")
        logger.warning(message)
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(f"::warning title=ECI bootstrap unavailable::{message}")
    return draws


def build(as_of: Optional[date] = None) -> dict:
    """Fetch, compute, and return the payload for data.json."""
    now = datetime.now(timezone.utc)
    models, report = load_models(decode(fetch(BENCHMARKED_MODELS_URL)))
    logger.info("%d scored models, %.0f%% with an Epoch open/closed label; using %d "
                "(%d left out without a label, %d without a date, %d without a country)",
                report.scored, 100 * report.coverage, len(models),
                len(report.left_out), len(report.undated), len(report.no_country))

    draws = load_draws(CACHE_DIR)
    if draws is not None:
        missing = sorted(m.name for m in models if not draws.has(m.name))
        if missing:
            logger.warning("%d model(s) are not in the bootstrap fit and use the analytical "
                           "test: %s", len(missing), missing[:10])

    payload = build_payload(models, as_of or now.date(), draws)
    return {
        "schema": payload.pop("schema"),
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        **payload,
    }


def to_json(payload: dict) -> str:
    """Indented JSON, except that a long list of numbers (the daily rolling
    average) stays on one line instead of taking a line per day, and the list
    of every model takes one line per model, so a new model is a one-line diff."""
    rows: dict[str, str] = {}

    def keep(text: str) -> str:
        token = f"@@row{len(rows)}@@"
        rows[token] = text
        return token

    def stash(node, key=None):
        if isinstance(node, dict):
            return {name: stash(value, name) for name, value in node.items()}
        if isinstance(node, list):
            if key == "models":
                return [keep(json.dumps(row, ensure_ascii=False)) for row in node]
            numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in node)
            if numbers and len(node) > 8:
                return keep(json.dumps(node))
            return [stash(value) for value in node]
        return node

    text = json.dumps(stash(payload), indent=2, ensure_ascii=False)
    # One pass, so text inside a row that has been put back is never looked at again.
    return re.sub(r'"(@@row\d+@@)"', lambda found: rows.get(found.group(1), found.group(0)), text) + "\n"


def write(payload: dict, path: Path) -> None:
    """Write atomically so a crash cannot leave half a file for the site."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(to_json(payload), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        payload = build()
        write(payload, DATA_JSON)
    except Exception:
        logger.exception("Update failed; data.json left as it was")
        return 1
    logger.info("Wrote %s", DATA_JSON.name)
    for name, comparison in payload["comparisons"].items():
        headline, today = comparison["headline"], comparison["today"]
        logger.info("  %s: %.1f months average (%s to %s), %.1f months today (%s vs %s)",
                    name, headline["months"], headline["window_start"], headline["window_end"],
                    today["months"], today["laggard_model"], today["reference_model"])
        trend = comparison["trend"]
        if trend:
            logger.info("  %s trendlines: %.1f months on %s, changing by %+.1f months a year",
                        name, trend["lag_months"], trend["on"], trend["change_months_per_year"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
