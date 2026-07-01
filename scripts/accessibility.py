"""Shared open/closed accessibility classification.

`classify_accessibility` maps an Epoch `Model accessibility` string to
True (open weights), False (closed: API/hosted), or None (unclassifiable —
callers must EXCLUDE such models from open-vs-closed analyses rather than
guess; see docs/audits/2026-07-01-calculation-logic-audit.md, findings F1/F2).

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

import pandas as pd

_CLOSED_VALUES = {"API access", "Hosted access (no API)"}


def classify_accessibility(value) -> bool | None:
    """True = open weights, False = closed (API/hosted), None = unclassifiable."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if "Open" in text:
        return True
    if text in _CLOSED_VALUES:
        return False
    return None


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
