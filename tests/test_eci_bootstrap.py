"""The paired-bootstrap wrapper: paired shares, fail-open build, and the cache."""
import numpy as np
import pytest

import eci_bootstrap as ebm
from eci_bootstrap import EciBootstrap, build_eci_bootstrap

CSV = b"model_id,benchmark_id,performance,benchmark,Model\nm1,b1,0.5,X,A\n"


def toy():
    draws = {
        "A": np.array([10.0, 11.0, 12.0, 9.0, 10.5]),
        "B": np.array([1.0, 2.0, 1.5, 0.5, 1.2]),
        "C": np.array([10.5, 10.0, 11.5, 9.5, 10.0]),
    }
    return EciBootstrap(draws, n_samples=5, seed=1, source_hash="h")


def test_prob_exceeds_compares_within_each_resample():
    b = toy()
    assert b.prob_exceeds("A", "B") == 1.0
    assert b.prob_exceeds("A", "C") == pytest.approx(0.6)   # resamples 2, 3 and 5
    assert b.prob_exceeds("B", "A") == 0.0


def test_prob_exceeds_is_none_for_a_model_missing_from_the_fit():
    b = toy()
    assert b.prob_exceeds("A", "ZZZ") is None
    assert b.prob_exceeds("ZZZ", "A") is None
    assert b.has("A") and not b.has("ZZZ")
    assert b.model_names == {"A", "B", "C"}


def test_prob_exceeds_skips_resamples_with_a_missing_draw():
    b = EciBootstrap({"A": np.array([2.0, np.nan, 2.0, 0.0]), "B": np.array([1.0, 1.0, np.nan, 1.0])},
                     n_samples=4, seed=1, source_hash="h")
    assert b.prob_exceeds("A", "B") == pytest.approx(0.5)   # resamples 1 and 4 are usable
    empty = EciBootstrap({"A": np.array([np.nan]), "B": np.array([1.0])},
                         n_samples=1, seed=1, source_hash="h")
    assert empty.prob_exceeds("A", "B") is None


def counting_fit(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(ebm, "_read_frame", lambda csv_bytes: "FRAME")

    def fake_fit(frame, n_samples, seed):
        calls["n"] += 1
        return ["A", "B"], np.array([[3.0, 1.0], [4.0, 1.0], [2.0, 5.0]])

    monkeypatch.setattr(ebm, "_fit_capability_draws", fake_fit)
    return calls


def test_build_returns_none_instead_of_raising(monkeypatch):
    def boom(frame, n_samples, seed):
        raise ValueError("fit failed")

    monkeypatch.setattr(ebm, "_read_frame", lambda csv_bytes: "FRAME")
    monkeypatch.setattr(ebm, "_fit_capability_draws", boom)
    assert build_eci_bootstrap(CSV) is None


def test_build_returns_none_when_the_source_lacks_required_columns():
    assert build_eci_bootstrap(b"model_id,performance\nm1,0.5\n") is None


def test_cache_round_trip_skips_the_refit(monkeypatch, tmp_path):
    calls = counting_fit(monkeypatch)
    first = build_eci_bootstrap(CSV, n_samples=3, cache_dir=tmp_path)
    assert calls["n"] == 1 and first.n_samples == 3
    assert first.prob_exceeds("A", "B") == pytest.approx(2 / 3)

    second = build_eci_bootstrap(CSV, n_samples=3, cache_dir=tmp_path)
    assert calls["n"] == 1                                    # loaded, not refitted
    assert second.prob_exceeds("A", "B") == pytest.approx(2 / 3)
    assert second.source_hash == first.source_hash


@pytest.mark.parametrize("change", ["source", "seed", "n_samples"])
def test_cache_is_ignored_when_its_inputs_change(monkeypatch, tmp_path, change):
    calls = counting_fit(monkeypatch)
    build_eci_bootstrap(CSV, n_samples=3, seed=1, cache_dir=tmp_path)
    kwargs = {"n_samples": 3, "seed": 1}
    source = CSV
    if change == "source":
        source = CSV + b"m2,b1,0.6,X,B\n"
    elif change == "seed":
        kwargs["seed"] = 2
    else:
        kwargs["n_samples"] = 5
    build_eci_bootstrap(source, cache_dir=tmp_path, **kwargs)
    assert calls["n"] == 2


def test_an_unreadable_cache_is_refitted(monkeypatch, tmp_path):
    calls = counting_fit(monkeypatch)
    (tmp_path / ebm.CACHE_FILENAME).write_bytes(b"not an npz file")
    assert build_eci_bootstrap(CSV, n_samples=3, cache_dir=tmp_path) is not None
    assert calls["n"] == 1
