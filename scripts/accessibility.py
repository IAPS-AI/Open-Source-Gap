"""Shared open/closed accessibility classification.

`classify_accessibility` maps an Epoch `Model accessibility` string to
True (open weights), False (closed: API, hosted, or limited access), or None
(unclassifiable — callers must EXCLUDE such models from open-vs-closed
analyses rather than guess; see
docs/audits/2026-07-01-calculation-logic-audit.md, findings F1/F2).

"Limited access" (Epoch label first used for Gemini 4 Argon, 2026-09) counts
as closed: the tracker compares open weights against everything else, and
limited-access frontier models are capability the closed side holds (METR's
Claude Mythos Preview is treated the same way). Any other label Epoch adds
is reported by `report_unrecognized_labels` and excluded until classified
here.

`ACCESSIBILITY_OVERRIDES` fills gaps where Epoch's data is blank or wrong.
Rows marked [ocg] are taken verbatim from
https://github.com/htihle/open_closed_gap `data_patches/
eci_accessibility_overrides.csv` (hand-verified by its author against public
release information). The remaining rows are this repo's additions for
`benchmarked_models.csv` display names that Epoch currently ships blank.
Keys cover both Epoch namespaces: index/benchmark "Model version" slugs and
benchmarked_models "Model" display names.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

_CLOSED_VALUES = {"API access", "Hosted access (no API)", "Limited access"}
# Labels excluded by design (no public release / status unknown), as opposed
# to labels this module has never seen.
_EXCLUDED_VALUES = {"Unreleased", "Unknown"}


class AccessibilitySourceError(RuntimeError):
    """Epoch's accessibility labels are missing or too sparse to trust.

    Raised instead of falling back to guesses: the 2026-09-05 build lost the
    label source silently and published heuristic open/closed flags for six
    weeks. update_data.py lets this propagate so the daily build fails and
    the previous data.json stays live."""


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def classify_accessibility(value) -> bool | None:
    """True = open weights, False = closed (API/hosted/limited),
    None = unclassifiable."""
    text = _clean(value)
    if not text:
        return None
    if "Open" in text:
        return True
    if text in _CLOSED_VALUES:
        return False
    return None


def is_unrecognized_label(value) -> bool:
    """True for a non-blank label with no rule here — a new Epoch category
    that needs a classification decision rather than silent exclusion."""
    text = _clean(value)
    return (bool(text) and classify_accessibility(text) is None
            and text not in _EXCLUDED_VALUES)


def report_unrecognized_labels(values: Iterable, source: str) -> list[str]:
    """Log (and annotate the GitHub Actions run with) any unrecognized
    accessibility labels in `values`. Returns them sorted."""
    unknown = sorted({_clean(v) for v in values if is_unrecognized_label(v)})
    if unknown:
        msg = (f"{source}: unrecognized Epoch accessibility label(s) {unknown}; "
               "models carrying them are EXCLUDED until scripts/accessibility.py "
               "classifies them")
        logger.warning(msg)
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(f"::warning title=Unrecognized accessibility label::{msg}")
    return unknown


ACCESSIBILITY_OVERRIDES: dict[str, str] = {
    # ---- [ocg] source repo's hand-verified overrides ----
    "Baichuan2-13B-Chat": "Open weights (unrestricted)",
    "CodeQwen1.5-7B": "Open weights (unrestricted)",
    "DeepSeek-Coder-V2-Lite-Base": "Open weights (unrestricted)",
    "DeepSeek-Coder-V2-Lite-Instruct": "Open weights (unrestricted)",
    "INTELLECT-1-Instruct": "Open weights (unrestricted)",
    "PaLM 62B": "Unreleased",
    "Qwen-1_8B": "Open weights (unrestricted)",
    "Qwen2.5-Coder-0.5B": "Open weights (unrestricted)",
    "Qwen2.5-Coder-14B": "Open weights (unrestricted)",
    "RedPajama-INCITE-7B-Base": "Open weights (unrestricted)",
    "Yi-9B": "Open weights (unrestricted)",
    "c4ai-command-r-08-2024": "Open weights (non-commercial)",
    "chatglm2-6b": "Open weights (unrestricted)",
    "chutes/GLM-4.5-FP8": "Open weights (unrestricted)",
    "chutes/Qwen3-Next-80B-A3B-Instruct": "Open weights (unrestricted)",
    "codex-mini-2025-05-16": "API access",
    "deepinfra/Qwen3-Next-80B-A3B-Instruct": "Open weights (unrestricted)",
    "gemini-3-deep-think-preview": "API access",
    "glm-5.1": "Open weights (unrestricted)",
    "grok-code-fast-1": "API access",
    "internlm-7b": "Open weights (unrestricted)",
    "internlm-chat-20b": "Open weights (unrestricted)",
    "mistral-medium-2508": "API access",
    "mistral-small-2402": "API access",
    "nvidia-nemotron-nano-9b-v2": "Open weights (unrestricted)",
    "o3-pro-2025-06-10_high": "API access",
    "o3-pro-2025-06-10_low": "API access",
    "o3-pro-2025-06-10_medium": "API access",
    "open_llama_7b": "Open weights (unrestricted)",
    "openhands-lm-32b-v0.1": "Open weights (unrestricted)",
    "opt-13b": "Open weights (unrestricted)",
    "qwen3-coder-next": "Open weights (unrestricted)",
    "qwen3.5-flash": "API access",
    "qwen3.5-plus": "API access",
    "qwen3.6-flash": "API access",
    "qwen3.6-max-preview": "API access",
    "text-davinci-003": "API access",
    # ---- this repo: WizardLM-2 (Microsoft; weights released 2024-04) ----
    "WizardLM-2-8x22B": "Open weights (unrestricted)",
    # ---- this repo: benchmarked_models.csv display names shipped blank ----
    "GLM-5.1": "Open weights (unrestricted)",
    "INTELLECT-1": "Open weights (unrestricted)",
    "Qwen 3.6 Max (Preview)": "API access",
    "Qwen 3.5 Plus (hosted 397B-A17B)": "API access",
    "Qwen 3.6 Flash": "API access",
    "Qwen 3.5 Flash (hosted 35B-A3B)": "API access",
    "o3-pro": "API access",
    "Gemini 3.5 Flash": "API access",
}
