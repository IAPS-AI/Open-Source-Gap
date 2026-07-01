"""Tests for the shared open/closed accessibility classification.

Run with: pytest tests/test_accessibility.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from accessibility import ACCESSIBILITY_OVERRIDES, classify_accessibility


class TestClassifyAccessibility:
    @pytest.mark.parametrize("value", [
        "Open weights (unrestricted)",
        "Open weights (restricted use)",
        "Open weights (non-commercial)",
    ])
    def test_open_weight_variants(self, value):
        assert classify_accessibility(value) is True

    @pytest.mark.parametrize("value", [
        "API access",
        "Hosted access (no API)",
    ])
    def test_closed_variants(self, value):
        assert classify_accessibility(value) is False

    @pytest.mark.parametrize("value", [
        None, "", "   ", float("nan"), "Unreleased", "Unknown",
        "some new label",
    ])
    def test_unclassifiable_returns_none(self, value):
        assert classify_accessibility(value) is None


class TestOverrides:
    def test_hosted_qwen_family_is_closed(self):
        for name in ("qwen3.5-plus", "qwen3.6-flash", "qwen3.6-max-preview",
                     "Qwen 3.6 Max (Preview)", "Qwen 3.6 Flash",
                     "Qwen 3.5 Plus (hosted 397B-A17B)",
                     "Qwen 3.5 Flash (hosted 35B-A3B)"):
            assert classify_accessibility(ACCESSIBILITY_OVERRIDES[name]) is False, name

    def test_known_open_models(self):
        for name in ("glm-5.1", "GLM-5.1", "WizardLM-2-8x22B", "INTELLECT-1",
                     "INTELLECT-1-Instruct"):
            assert classify_accessibility(ACCESSIBILITY_OVERRIDES[name]) is True, name

    def test_blank_eci_closed_models_covered(self):
        for name in ("o3-pro", "Gemini 3.5 Flash", "text-davinci-003",
                     "PaLM 62B"):
            assert name in ACCESSIBILITY_OVERRIDES
        assert classify_accessibility(ACCESSIBILITY_OVERRIDES["o3-pro"]) is False
        # Unreleased models classify to None (excluded), not closed.
        assert classify_accessibility(ACCESSIBILITY_OVERRIDES["PaLM 62B"]) is None


class TestFetcherClassificationChain:
    """csv_benchmark_fetcher._classify_model precedence:
    CSV accessibility column > Epoch index > overrides > heuristics > None."""

    def _fetcher(self, index_map):
        from csv_benchmark_fetcher import CSVBenchmarkFetcher
        f = CSVBenchmarkFetcher()
        f._accessibility_by_version = index_map
        return f

    def _row(self, name, org="SomeOrg", acc=None):
        row = {"Model version": name, "Organization": org}
        if acc is not None:
            row["Model accessibility"] = acc
        return pd.Series(row)

    def test_csv_column_takes_precedence(self):
        f = self._fetcher({"m": "API access"})
        assert f._classify_model(self._row("m", acc="Open weights (unrestricted)")) is True

    def test_index_map_used_when_no_csv_column(self):
        f = self._fetcher({"mistral-large-2402": "API access"})
        assert f._classify_model(self._row("mistral-large-2402", org="Mistral AI")) is False

    def test_override_fills_index_blank(self):
        f = self._fetcher({})
        assert f._classify_model(self._row("qwen3.5-plus", org="Alibaba")) is False
        assert f._classify_model(self._row("glm-5.1", org="Z.ai")) is True

    def test_heuristic_last_resort_unambiguous_only(self):
        f = self._fetcher({})
        assert f._classify_model(self._row("Meta-Llama-3-70B-Instruct", org="Meta")) is True
        assert f._classify_model(self._row("gpt-4-0314", org="OpenAI")) is False
        # gpt-oss is open despite the OpenAI org
        assert f._classify_model(self._row("openai/gpt-oss-120b_high", org="OpenAI")) is True
        # mixed labs are no longer guessed by org alone
        assert f._classify_model(self._row("qwen-mystery-model", org="Alibaba")) is None
        assert f._classify_model(self._row("mistral-mystery", org="Mistral AI")) is None
        assert f._classify_model(self._row("phi-9", org="Microsoft")) is None

    def test_unknown_model_returns_none(self):
        f = self._fetcher({})
        assert f._classify_model(self._row("totally-unknown", org="NewLab")) is None
