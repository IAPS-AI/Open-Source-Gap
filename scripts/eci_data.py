"""Load Epoch AI's ECI scores into `gap.Model` records.

Source: https://epoch.ai/data/benchmarked_models.csv. It has one row per
benchmarked model *version* (GPT-5 at each reasoning effort, a hosted copy of
open weights, ...), but the ECI is fitted per `model_group`, so that is the
unit here. Every version in a group carries the same score, label, and
organization; the group's release date is its earliest version's. Dated
releases of one family, such as GPT-4o (May 2024) and GPT-4o (Nov 2024), are
separate groups with separate scores, and each counts on its own date.

Open or closed comes from Epoch's `Model accessibility` label and nothing
else (see `accessibility.py`). Country comes from Epoch's `Country (of
organization)` label, which is where the developer is based.
"""
from __future__ import annotations

import csv
import io
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from accessibility import (
    ACCESSIBILITY_OVERRIDES,
    MIN_LABEL_COVERAGE,
    AccessibilitySourceError,
    classify_accessibility,
    report_unrecognized_labels,
)
from gap import Model

logger = logging.getLogger(__name__)

BENCHMARKED_MODELS_URL = "https://epoch.ai/data/benchmarked_models.csv"
ECI_BENCHMARKS_URL = "https://epoch.ai/data/eci_benchmarks.csv"

# epoch.ai answers Python's default User-Agent with 403.
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

LABEL_COLUMN = "Model accessibility"
COUNTRY_COLUMN = "Country (of organization)"
REQUIRED_COLUMNS = ("model_group", "eci", "Version release date", "Organization", COUNTRY_COLUMN)


def fetch(url: str, timeout: int = 60, attempts: int = 3) -> bytes:
    """GET `url`, retrying a failed attempt twice."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == attempts:
                raise
            logger.warning("Fetching %s failed (%s); retrying", url, error)
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def decode(raw: bytes) -> str:
    """Epoch serves UTF-8 without declaring it."""
    return raw.decode("utf-8-sig", errors="replace")


@dataclass
class LoadReport:
    scored: int = 0                                  # groups with an ECI score
    labelled: int = 0                                # ... and an Epoch open/closed label
    left_out: list[str] = field(default_factory=list)   # no usable label
    undated: list[str] = field(default_factory=list)    # no release date
    no_country: list[str] = field(default_factory=list) # used, but in neither country group

    @property
    def coverage(self) -> float:
        return self.labelled / self.scored if self.scored else 0.0


def _number(text: Optional[str]) -> Optional[float]:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value == value else None  # drop NaN


def _day(text: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat((text or "").strip()[:10])
    except ValueError:
        return None


def _group_label(labels: list[str]) -> str:
    """One label for a group. Its versions agree in practice; if they ever
    differ, published weights make the model open whatever else is offered."""
    known = [label for label in labels if classify_accessibility(label) is not None]
    for label in known:
        if classify_accessibility(label):
            return label
    if known:
        return known[0]
    return next((label for label in labels if label), "")


def _first(rows: list[dict], column: str) -> str:
    return next((value for value in ((r.get(column) or "").strip() for r in rows) if value), "")


def load_models(csv_text: str) -> tuple[list[Model], LoadReport]:
    """Parse benchmarked_models.csv into one `Model` per scored `model_group`.

    Raises AccessibilitySourceError when the label column is missing or fewer
    than 85% of scored models carry an Epoch open/closed label, and ValueError
    when another required column is missing, nothing is scored, or fewer than
    85% of the models used carry a country."""
    csv.field_size_limit(10_000_000)  # descriptions are long, multi-line fields
    reader = csv.DictReader(io.StringIO(csv_text))
    columns = set(reader.fieldnames or [])
    if LABEL_COLUMN not in columns:
        raise AccessibilitySourceError(
            f"benchmarked_models.csv has no '{LABEL_COLUMN}' column; "
            "refusing to classify open or closed without Epoch's labels")
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"benchmarked_models.csv is missing columns {missing}")

    groups: dict[str, list[dict]] = {}
    for row in reader:
        if _number(row.get("eci")) is None:
            continue  # not scored (yet)
        name = (row.get("model_group") or "").strip() or (row.get("Model") or "").strip()
        if name:
            groups.setdefault(name, []).append(row)
    if not groups:
        raise ValueError("benchmarked_models.csv has no rows with an ECI score")

    report = LoadReport(scored=len(groups))
    models: list[Model] = []
    epoch_labels: list[str] = []
    for name, rows in groups.items():
        epoch_label = _group_label([(r.get(LABEL_COLUMN) or "").strip() for r in rows])
        epoch_labels.append(epoch_label)
        if classify_accessibility(epoch_label) is not None:
            report.labelled += 1
        is_open = classify_accessibility(epoch_label or ACCESSIBILITY_OVERRIDES.get(name))
        if is_open is None:
            report.left_out.append(name)
            continue

        days = [d for d in (_day(r.get("Version release date")) for r in rows) if d]
        if not days:
            report.undated.append(name)
            continue

        best = max(rows, key=lambda r: _number(r["eci"]))
        models.append(Model(
            name=name,
            released=min(days),
            eci=_number(best["eci"]),
            is_open=is_open,
            ci_low=_number(best.get("eci_ci_low")),
            ci_high=_number(best.get("eci_ci_high")),
            organization=_first(rows, "Organization"),
            country=_first(rows, COUNTRY_COLUMN),
        ))

    report_unrecognized_labels(epoch_labels, "benchmarked_models.csv")
    if report.coverage < MIN_LABEL_COVERAGE:
        raise AccessibilitySourceError(
            f"only {report.coverage:.1%} of ECI-scored models carry an Epoch open/closed "
            f"label (need {MIN_LABEL_COVERAGE:.0%}); refusing to fill the rest with guesses")
    if report.undated:
        logger.warning("Left out %d scored model(s) with no release date: %s",
                       len(report.undated), report.undated)
    report.no_country = [m.name for m in models if not m.country]
    if len(report.no_country) > (1 - MIN_LABEL_COVERAGE) * len(models):
        raise ValueError(
            f"{len(report.no_country)} of {len(models)} models have no '{COUNTRY_COLUMN}'; "
            "refusing to compare countries on a partial list")
    models.sort(key=lambda m: (m.released, m.name))
    return models, report
