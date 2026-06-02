"""Refit the ECI IRT model with bootstrap (via eci-public) and expose paired
capability draws for the gap "caught up" criterion.

Displayed ECI scores come from Epoch's published benchmarked_models.csv; this
module supplies ONLY the joint bootstrap draws used to decide matches. Every
public entry point is fail-open: build_eci_bootstrap returns None on any
failure so the daily pipeline reverts to the analytical criterion.
"""
from __future__ import annotations

import hashlib
import logging
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ECI_BENCHMARKS_URL = "https://epoch.ai/data/eci_benchmarks.csv"
BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_SEED = 12345
CACHE_FILENAME = "eci_bootstrap_cache.npz"
# epoch.ai blocks the default urllib User-Agent with 403.
EPOCH_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQUIRED_COLUMNS = {"model_id", "benchmark_id", "performance", "benchmark", "Model"}


class EciBootstrap:
    """Joint bootstrap capability draws keyed by Epoch ``Model`` name.

    ``draws[name]`` is a 1-D array of length B; ``draws[a][i]`` and
    ``draws[b][i]`` are the same resample i, so comparisons are paired.
    """

    def __init__(self, draws: dict[str, np.ndarray], *, n_samples: int,
                 seed: int, source_hash: str):
        self.draws = draws
        self.n_samples = n_samples
        self.seed = seed
        self.source_hash = source_hash

    def has(self, name: str) -> bool:
        return name in self.draws

    @property
    def model_names(self) -> set[str]:
        return set(self.draws.keys())

    def prob_exceeds(self, a: str, b: str) -> Optional[float]:
        """P(capability_a > capability_b) across paired resamples, or None if
        either model is absent from the fit."""
        da = self.draws.get(a)
        db = self.draws.get(b)
        if da is None or db is None:
            return None
        n = min(len(da), len(db))
        if n == 0:
            return None
        return float((da[:n] > db[:n]).mean())
