"""The entry point: fetch, build, write, and fail without touching data.json."""
import json
from datetime import date

import pytest

import update_data
from eci_data import BENCHMARKED_MODELS_URL, ECI_BENCHMARKS_URL
from test_eci_data import row, sheet

MODELS_CSV = sheet(
    row("Closed A", 120, "2024-01-01"),
    row("Closed B", 131.5, "2025-03-01"),
    row("Open A", 119.5, "2024-06-01", label="Open weights (unrestricted)", org="Tülu Lab", country="China"),
    row("Open B", 131, "2025-08-01", label="Open weights (unrestricted)", country="China"),
).encode("utf-8")


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No network, no real bootstrap, and data.json in a temp directory."""
    served = {BENCHMARKED_MODELS_URL: MODELS_CSV, ECI_BENCHMARKS_URL: b"eci benchmarks"}
    monkeypatch.setattr(update_data, "fetch", lambda url, **kwargs: served[url])
    monkeypatch.setattr(update_data, "build_eci_bootstrap", lambda csv_bytes, cache_dir=None: None)
    monkeypatch.setattr(update_data, "DATA_JSON", tmp_path / "data.json")
    monkeypatch.setattr(update_data, "CACHE_DIR", tmp_path / "cache")
    return served


def test_build_returns_the_payload_stamped_with_the_build_time(offline):
    payload = update_data.build(as_of=date(2025, 9, 1))
    assert list(payload)[:3] == ["schema", "generated_at", "as_of"]
    assert list(payload)[-1] == "models"                    # the long list goes last in the file
    assert [m["model"] for m in payload["models"]] == ["Open B", "Closed B", "Open A", "Closed A"]
    assert payload["as_of"] == "2025-09-01"
    assert payload["generated_at"].endswith("Z") and len(payload["generated_at"]) == 20
    assert list(payload["comparisons"]) == ["open_closed", "china_us"]
    for comparison in payload["comparisons"].values():
        assert comparison["today"]["laggard_model"] == "Open B"
        assert comparison["today"]["reference_model"] == "Closed B"
    assert payload["test"]["paired_bootstrap"] is False


def test_main_writes_readable_utf8_json(offline):
    assert update_data.main() == 0
    text = update_data.DATA_JSON.read_text(encoding="utf-8")
    assert text.endswith("\n") and "Tülu Lab" in text
    payload = json.loads(text)
    assert payload["schema"] == 2
    assert len(payload["comparisons"]["open_closed"]["leader_frontier"]) == 2
    assert not update_data.DATA_JSON.with_suffix(".json.tmp").exists()
    # One line per model, so a new model is a one-line change in the daily commit.
    rows = [line.strip().rstrip(",") for line in text.splitlines() if line.lstrip().startswith('{"model"')]
    assert [json.loads(row) for row in rows] == payload["models"] and len(rows) == 4
    assert '"organization": "Tülu Lab", "country": "China", "open": true}' in rows[2]


def test_long_runs_of_numbers_stay_on_one_line():
    text = update_data.to_json({"months": [0.5] * 9, "short": [1, 2], "flags": [True] * 9,
                                "models": [{"model": "A @@row0@@"}, {"model": "B"}]})
    assert json.loads(text) == {"months": [0.5] * 9, "short": [1, 2], "flags": [True] * 9,
                                "models": [{"model": "A @@row0@@"}, {"model": "B"}]}
    lines = text.splitlines()
    assert '  "months": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],' in lines
    assert '    {"model": "B"}' in lines
    assert len([line for line in lines if line.strip() in ("true,", "true")]) == 9    # not numbers


def test_main_leaves_data_json_alone_when_the_labels_are_missing(offline):
    update_data.DATA_JSON.write_text("previous good data", encoding="utf-8")
    offline[BENCHMARKED_MODELS_URL] = MODELS_CSV.replace(b"Model accessibility", b"Something else")
    assert update_data.main() == 1
    assert update_data.DATA_JSON.read_text(encoding="utf-8") == "previous good data"


def test_main_leaves_data_json_alone_when_the_fetch_fails(offline, monkeypatch):
    def down(url, **kwargs):
        raise OSError("network down")

    update_data.DATA_JSON.write_text("previous good data", encoding="utf-8")
    monkeypatch.setattr(update_data, "fetch", down)
    assert update_data.main() == 1
    assert update_data.DATA_JSON.read_text(encoding="utf-8") == "previous good data"


def test_a_missing_bootstrap_is_a_visible_warning_not_a_failure(offline, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert update_data.load_draws(cache_dir=None) is None
    assert "::warning title=ECI bootstrap unavailable::" in capsys.readouterr().out


def test_a_failed_benchmark_fetch_also_falls_back(offline, monkeypatch):
    def partial(url, **kwargs):
        if url == ECI_BENCHMARKS_URL:
            raise OSError("404")
        return offline[url]

    monkeypatch.setattr(update_data, "fetch", partial)
    assert update_data.load_draws(cache_dir=None) is None
    assert update_data.main() == 0


def test_draws_reach_the_payload_when_the_bootstrap_builds(offline, monkeypatch):
    class Draws:
        n_samples = 500

        def has(self, name):
            return True

        def prob_exceeds(self, a, b):
            return 1.0 if (a, b) == ("Open B", "Closed B") else 0.0

    monkeypatch.setattr(update_data, "build_eci_bootstrap", lambda csv_bytes, cache_dir=None: Draws())
    payload = update_data.build(as_of=date(2025, 9, 1))
    assert payload["test"] == {"paired_bootstrap": True, "resamples": 500, "min_share": 0.05}
    assert payload["comparisons"]["china_us"]["leader_frontier"][1]["test"] == "bootstrap"
