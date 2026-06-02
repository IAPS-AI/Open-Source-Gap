"""Tests for the eci-public bootstrap wrapper."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from eci_bootstrap import EciBootstrap


def _toy():
    # A: always above B, sometimes above C. B below everyone.
    draws = {
        "A": np.array([10.0, 11.0, 12.0, 9.0, 10.5]),
        "B": np.array([1.0, 2.0, 1.5, 0.5, 1.2]),
        "C": np.array([10.5, 10.0, 11.5, 9.5, 10.0]),
    }
    return EciBootstrap(draws, n_samples=5, seed=1, source_hash="h")


def test_prob_exceeds_basic():
    b = _toy()
    assert b.prob_exceeds("A", "B") == 1.0           # A > B in all 5
    # A>C in samples where 10>10.5? no; 11>10 yes; 12>11.5 yes; 9>9.5 no; 10.5>10 yes -> 3/5
    assert b.prob_exceeds("A", "C") == pytest.approx(0.6)
    assert b.prob_exceeds("B", "A") == 0.0


def test_prob_exceeds_missing_returns_none():
    b = _toy()
    assert b.prob_exceeds("A", "ZZZ") is None
    assert b.prob_exceeds("ZZZ", "A") is None


def test_has_and_model_names():
    b = _toy()
    assert b.has("A") and not b.has("ZZZ")
    assert b.model_names == {"A", "B", "C"}


import eci_bootstrap as ebm


def test_build_failopen_on_fetch_error(monkeypatch):
    def boom(url, timeout=120):
        raise RuntimeError("network down")
    monkeypatch.setattr(ebm, "_fetch_csv_text", boom)
    assert ebm.build_eci_bootstrap(cache_dir=None) is None


def test_build_failopen_on_fit_error(monkeypatch):
    monkeypatch.setattr(ebm, "_fetch_csv_text",
                        lambda url, timeout=120: "model_id,benchmark_id,performance,benchmark,Model\n")
    monkeypatch.setattr(ebm, "_read_validated_df", lambda text: object())

    def boom(df, n_samples, seed, use_analytical_jacobian):
        raise ValueError("fit failed")
    monkeypatch.setattr(ebm, "_fit_capability_draws", boom)
    assert ebm.build_eci_bootstrap(cache_dir=None) is None


def test_build_and_cache_roundtrip(monkeypatch, tmp_path):
    text = "model_id,benchmark_id,performance,benchmark,Model\nm1,b1,0.5,X,A\n"
    monkeypatch.setattr(ebm, "_fetch_csv_text", lambda url, timeout=120: text)
    monkeypatch.setattr(ebm, "_read_validated_df", lambda t: "DF")

    calls = {"n": 0}

    def fake_fit(df, n_samples, seed, use_analytical_jacobian):
        calls["n"] += 1
        names = ["A", "B"]
        matrix = np.array([[3.0, 1.0], [4.0, 1.0], [2.0, 5.0]])  # (B=3, 2 models)
        return names, matrix
    monkeypatch.setattr(ebm, "_fit_capability_draws", fake_fit)

    b1 = ebm.build_eci_bootstrap(n_samples=3, cache_dir=str(tmp_path))
    assert b1 is not None
    assert calls["n"] == 1
    assert b1.prob_exceeds("A", "B") == pytest.approx(2 / 3)  # 3>1,4>1,2<5

    # Second call with same input hash loads cache, does NOT refit.
    b2 = ebm.build_eci_bootstrap(n_samples=3, cache_dir=str(tmp_path))
    assert b2 is not None
    assert calls["n"] == 1
    assert b2.prob_exceeds("A", "B") == pytest.approx(2 / 3)


def _patch_counting_fit(monkeypatch, calls):
    monkeypatch.setattr(ebm, "_read_validated_df", lambda t: "DF")

    def fake_fit(df, n_samples, seed, use_analytical_jacobian):
        calls["n"] += 1
        return ["A", "B"], np.array([[3.0, 1.0], [4.0, 1.0], [2.0, 5.0]])
    monkeypatch.setattr(ebm, "_fit_capability_draws", fake_fit)


def test_cache_invalidates_on_source_change(monkeypatch, tmp_path):
    calls = {"n": 0}
    _patch_counting_fit(monkeypatch, calls)
    monkeypatch.setattr(ebm, "_fetch_csv_text", lambda url, timeout=120: "TEXT-A")
    ebm.build_eci_bootstrap(n_samples=3, cache_dir=str(tmp_path))
    ebm.build_eci_bootstrap(n_samples=3, cache_dir=str(tmp_path))
    assert calls["n"] == 1  # same hash -> cache hit
    monkeypatch.setattr(ebm, "_fetch_csv_text", lambda url, timeout=120: "TEXT-B")
    ebm.build_eci_bootstrap(n_samples=3, cache_dir=str(tmp_path))
    assert calls["n"] == 2  # different source -> refit


def test_cache_invalidates_on_seed_change(monkeypatch, tmp_path):
    calls = {"n": 0}
    _patch_counting_fit(monkeypatch, calls)
    monkeypatch.setattr(ebm, "_fetch_csv_text", lambda url, timeout=120: "TEXT")
    ebm.build_eci_bootstrap(n_samples=3, seed=1, cache_dir=str(tmp_path))
    ebm.build_eci_bootstrap(n_samples=3, seed=2, cache_dir=str(tmp_path))
    assert calls["n"] == 2  # different seed -> refit


def test_cache_invalidates_on_nsamples_change(monkeypatch, tmp_path):
    calls = {"n": 0}
    _patch_counting_fit(monkeypatch, calls)
    monkeypatch.setattr(ebm, "_fetch_csv_text", lambda url, timeout=120: "TEXT")
    ebm.build_eci_bootstrap(n_samples=3, cache_dir=str(tmp_path))
    ebm.build_eci_bootstrap(n_samples=5, cache_dir=str(tmp_path))
    assert calls["n"] == 2  # different requested sample count -> refit
