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
