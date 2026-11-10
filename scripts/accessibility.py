"""Open or closed? Epoch's `Model accessibility` label decides, never a guess.

`classify_accessibility` maps a label to True (open weights), False (closed:
API, hosted, or limited access), or None (no public release, unknown, or a
label this module has not seen). A model that comes back None is left out of
the analysis: guessing a side moves the frontier.

"Limited access" counts as closed. The tracker compares open weights against
everything else, and a limited-access frontier model is capability the closed
side holds.

`ACCESSIBILITY_OVERRIDES` fills labels Epoch ships blank, keyed by the model
name the ECI uses. It never replaces a label Epoch does ship.

If the labels go missing or thin out, `AccessibilitySourceError` stops the
build so the previous data.json stays live. A build in September 2026 lost the
label source quietly and published guessed flags for six weeks.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

CLOSED_LABELS = {"API access", "Hosted access (no API)", "Limited access"}
# Left out by design, as opposed to labels this module has never seen.
EXCLUDED_LABELS = {"Unreleased", "Unknown"}

# Below this share of scored models carrying an Epoch open/closed label, the
# source is treated as broken rather than filled in from the overrides.
MIN_LABEL_COVERAGE = 0.85


class AccessibilitySourceError(RuntimeError):
    """Epoch's accessibility labels are missing or too sparse to trust."""


def _clean(label) -> str:
    if label is None or label != label:  # None or NaN
        return ""
    return str(label).strip()


def classify_accessibility(label) -> Optional[bool]:
    """True = open weights, False = closed, None = leave the model out."""
    text = _clean(label)
    if not text:
        return None
    if "Open" in text:
        return True
    if text in CLOSED_LABELS:
        return False
    return None


def is_unrecognized_label(label) -> bool:
    """A non-blank label with no rule here: a new Epoch category that needs a
    decision, not quiet exclusion."""
    text = _clean(label)
    return bool(text) and classify_accessibility(text) is None and text not in EXCLUDED_LABELS


def report_unrecognized_labels(labels: Iterable, source: str) -> list[str]:
    """Log any unrecognized labels and flag them on the GitHub Actions run."""
    unknown = sorted({_clean(v) for v in labels if is_unrecognized_label(v)})
    if unknown:
        message = (f"{source}: unrecognized Epoch accessibility label(s) {unknown}; models "
                   "carrying them are left out until scripts/accessibility.py classifies them")
        logger.warning(message)
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(f"::warning title=Unrecognized accessibility label::{message}")
    return unknown


ACCESSIBILITY_OVERRIDES: dict[str, str] = {
    # Blank in benchmarked_models.csv as of November 2026.
    "INTELLECT-1": "Open weights (unrestricted)",
    "Qwen 3.5 Flash (hosted 35B-A3B)": "API access",
    "Qwen 3.5 Plus (hosted 397B-A17B)": "API access",
    "Qwen 3.6 Flash": "API access",
    "Qwen 3.6 Max (Preview)": "API access",
    # Labelled now, shipped blank before: kept in case the label drops again.
    "GLM-5.1": "Open weights (unrestricted)",
    "Gemini 3.5 Flash": "API access",
    "o3-pro": "API access",
}
