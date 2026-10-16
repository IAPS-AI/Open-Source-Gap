"""Tests for the shared open/closed accessibility classification.

Run with: pytest tests/test_accessibility.py -v
"""

import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from accessibility import (
    ACCESSIBILITY_OVERRIDES,
    AccessibilitySourceError,
    classify_accessibility,
    is_unrecognized_label,
    report_unrecognized_labels,
)


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
        "Limited access",  # Gemini 4 Argon, 2026-09-30
    ])
    def test_closed_variants(self, value):
        assert classify_accessibility(value) is False

    @pytest.mark.parametrize("value", [
        None, "", "   ", float("nan"), "Unreleased", "Unknown",
        "some new label",
    ])
    def test_unclassifiable_returns_none(self, value):
        assert classify_accessibility(value) is None


class TestUnrecognizedLabels:
    @pytest.mark.parametrize("value", ["some new label", "Partner access"])
    def test_new_labels_flagged(self, value):
        assert is_unrecognized_label(value) is True

    @pytest.mark.parametrize("value", [
        None, "", "  ", float("nan"), "Unreleased", "Unknown", "API access",
        "Limited access", "Open weights (unrestricted)",
    ])
    def test_known_or_blank_labels_not_flagged(self, value):
        assert is_unrecognized_label(value) is False

    def test_report_dedupes_and_annotates_github_actions(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        found = report_unrecognized_labels(
            ["API access", "Partner access", "Partner access ", "Beta", None],
            "test.csv")
        assert found == ["Beta", "Partner access"]
        assert "::warning title=Unrecognized accessibility label::" in capsys.readouterr().out

    def test_report_silent_when_all_known(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        assert report_unrecognized_labels(["API access", "Unknown", ""], "t") == []
        assert capsys.readouterr().out == ""


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
    CSV accessibility column > Epoch model metadata > overrides > None."""

    def _fetcher(self, index_map):
        from csv_benchmark_fetcher import CSVBenchmarkFetcher
        f = CSVBenchmarkFetcher()
        f._accessibility_by_version = index_map
        f._display_name_by_version = {}
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

    def test_no_org_or_name_heuristics(self):
        # Unlabelled models are excluded whatever the org or name suggests:
        # "org == meta → open" counted Meta's API-only Muse line as open.
        f = self._fetcher({})
        for name, org in [("muse-wonder-1.0_max", "Meta AI"),
                          ("Meta-Llama-3-70B-Instruct", "Meta"),
                          ("gpt-4-0314", "OpenAI"),
                          ("openai/gpt-oss-120b_high", "OpenAI"),
                          ("gemini-9-mystery", "Google DeepMind"),
                          ("qwen-mystery-model", "Alibaba")]:
            assert f._classify_model(self._row(name, org=org)) is None, name

    def test_epoch_label_beats_org(self):
        f = self._fetcher({"muse-spark-1.2_xhigh": "API access",
                           "Muse-Spark-Open-816B-A42B_xhigh": "Open weights (restricted use)",
                           "gemini-4-argon_high": "Limited access"})
        assert f._classify_model(self._row("muse-spark-1.2_xhigh", org="Meta AI")) is False
        assert f._classify_model(self._row("Muse-Spark-Open-816B-A42B_xhigh", org="Meta AI")) is True
        assert f._classify_model(self._row("gemini-4-argon_high", org="Google DeepMind")) is False

    def test_unknown_model_returns_none(self):
        f = self._fetcher({})
        assert f._classify_model(self._row("totally-unknown", org="NewLab")) is None


METADATA_HEADER = ("model_version,model_group,date,display_name,organization,"
                   "country,accessibility,training_compute_flop\n")
BENCH_HEADER = "Model version,mean_score,Release date,Organization,Country,stderr\n"


def _zip_bytes(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


def _fetcher_from_zip(files: dict):
    from csv_benchmark_fetcher import CSVBenchmarkFetcher
    f = CSVBenchmarkFetcher()
    f._zip_data = _zip_bytes(files)
    return f


class TestModelMetadataSource:
    """The fetcher reads labels and display names from model_metadata.csv
    and refuses to run without them."""

    METADATA = METADATA_HEADER + (
        "gemini-4-argon_high,Gemini 4 Argon,2026-09-30,Gemini 4 Argon (high),Google DeepMind,US,Limited access,\n"
        "muse-wonder-1.0_max,Muse Wonder,2026-10-12,Muse Wonder (max),Meta AI,US,API access,\n"
        "kimi-k3_max,Kimi K3,2026-07-16,,Moonshot,China,Open weights (non-commercial),\n"
        "kimi-k3_max,Kimi K3,2026-07-16,Dup Row,Moonshot,China,API access,\n"
        ",,,,,,,\n"
    )
    BENCH = BENCH_HEADER + (
        "gemini-4-argon_high,0.964,2026-09-30,Google DeepMind,US,0.012\n"
        "muse-wonder-1.0_max,0.95,2026-10-12,Meta AI,US,0.01\n"
        "kimi-k3_max,0.90,2026-07-16,Moonshot,China,0.01\n"
    )

    def test_classification_and_display_names(self):
        f = _fetcher_from_zip({"model_metadata.csv": self.METADATA,
                               "gpqa_diamond.csv": self.BENCH})
        models = {m["model"]: m for m in f.fetch_benchmark_data("gpqa_diamond")["models"]}
        assert models["gemini-4-argon_high"]["is_open"] is False
        assert models["gemini-4-argon_high"]["display_name"] == "Gemini 4 Argon (high)"
        assert models["muse-wonder-1.0_max"]["is_open"] is False
        # first row wins for duplicated slugs; blank display name → slug
        assert models["kimi-k3_max"]["is_open"] is True
        assert models["kimi-k3_max"]["display_name"] == "kimi-k3_max"

    def test_missing_metadata_file_raises(self):
        f = _fetcher_from_zip({"gpqa_diamond.csv": self.BENCH,
                               "epoch_capabilities_index/eci_scores.csv": "Model,eci\n"})
        with pytest.raises(AccessibilitySourceError, match="model_metadata.csv"):
            f.fetch_benchmark_data("gpqa_diamond")

    def test_missing_accessibility_column_raises(self):
        meta = "model_version,display_name\ngemini-4-argon_high,Gemini 4 Argon (high)\n"
        f = _fetcher_from_zip({"model_metadata.csv": meta,
                               "gpqa_diamond.csv": self.BENCH})
        with pytest.raises(AccessibilitySourceError, match="accessibility"):
            f.fetch_benchmark_data("gpqa_diamond")

    def test_low_label_coverage_raises(self):
        # 1 of 3 scored rows carries an Epoch label; an override-covered row
        # (qwen3.5-plus) does not count toward Epoch coverage.
        meta = METADATA_HEADER + "kimi-k3_max,Kimi K3,2026-07-16,,Moonshot,China,Open weights (unrestricted),\n"
        bench = BENCH_HEADER + (
            "kimi-k3_max,0.90,2026-07-16,Moonshot,China,0.01\n"
            "qwen3.5-plus,0.88,2026-02-16,Alibaba,China,0.01\n"
            "mystery-model,0.80,2026-05-01,NewLab,US,0.01\n"
        )
        f = _fetcher_from_zip({"model_metadata.csv": meta, "gpqa_diamond.csv": bench})
        with pytest.raises(AccessibilitySourceError, match="33.3%"):
            f.fetch_benchmark_data("gpqa_diamond")


class TestBuildFailsWithoutLabels:
    """AccessibilitySourceError must escape process_all_benchmarks' per-
    benchmark error handling so main() exits 1 and CI skips the commit."""

    def _patch_pipeline(self, monkeypatch):
        import update_data

        class RaisingFetcher:
            def fetch_benchmark_data(self, benchmark_id):
                raise AccessibilitySourceError("labels gone")

        monkeypatch.setattr(update_data, "process_data", lambda: defaultdict(lambda: None))
        monkeypatch.setattr(update_data, "METR_FETCHER_AVAILABLE", False)
        monkeypatch.setattr(update_data, "CSV_FETCHER_AVAILABLE", True)
        monkeypatch.setattr(update_data, "CSVBenchmarkFetcher", RaisingFetcher)
        return update_data

    def test_process_all_benchmarks_propagates(self, monkeypatch):
        update_data = self._patch_pipeline(monkeypatch)
        with pytest.raises(AccessibilitySourceError):
            update_data.process_all_benchmarks()

    def test_main_exits_nonzero_without_writing(self, monkeypatch, tmp_path):
        update_data = self._patch_pipeline(monkeypatch)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc:
            update_data.main()
        assert exc.value.code == 1
        assert not (tmp_path / "data.json").exists()
