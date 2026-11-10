"""Paired bootstrap draws of model capability, for the "caught up" test.

Epoch decides whether an open model has caught up to a closed one from a
paired bootstrap: resample the benchmark scores with replacement, refit the
ECI on each resample, and compare the two models within each resample. Epoch
publishes the benchmark scores (eci_benchmarks.csv) and the fitting code
(https://github.com/epoch-research/eci-public), so this module reruns the fit
and keeps the draws.

Only the draws are used. Displayed scores stay Epoch's published ones, and a
comparison within one resample does not depend on the ECI's scale, so raw
capability draws are enough.

`build_eci_bootstrap` returns None if anything goes wrong (package not
installed, schema change, a fit that fails); the caller then falls back to
the analytical test and says so. The refit takes minutes, so the draws are
cached under a hash of the source file.
"""
from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_SEED = 12345
CACHE_FILENAME = "eci_bootstrap_cache.npz"
REQUIRED_COLUMNS = {"model_id", "benchmark_id", "performance", "benchmark", "Model"}


class EciBootstrap:
    """Capability draws keyed by ECI model name. `draws[a][i]` and
    `draws[b][i]` come from the same resample, so comparisons are paired."""

    def __init__(self, draws: dict[str, np.ndarray], *, n_samples: int, seed: int,
                 source_hash: str):
        self.draws = draws
        self.n_samples = n_samples
        self.seed = seed
        self.source_hash = source_hash

    def has(self, name: str) -> bool:
        return name in self.draws

    @property
    def model_names(self) -> set[str]:
        return set(self.draws)

    def prob_exceeds(self, a: str, b: str) -> Optional[float]:
        """Share of resamples in which `a` scores above `b`, or None if either
        model is missing from the fit."""
        x, y = self.draws.get(a), self.draws.get(b)
        if x is None or y is None:
            return None
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        usable = np.isfinite(x) & np.isfinite(y)  # a NaN would count as "not above"
        if not usable.any():
            return None
        return float((x[usable] > y[usable]).mean())


def _fit_capability_draws(frame, n_samples: int, seed: int):
    """Run eci-public's bootstrap fit: (model names, matrix[resample, model])."""
    from eci import fit_eci_model

    _, _, bootstrap = fit_eci_model(
        frame,
        bootstrap_samples=n_samples,
        bootstrap_seed=seed,
        use_analytical_jacobian=True,
        return_bootstrap_samples=True,
    )
    names = list(bootstrap["model_names"])
    samples = bootstrap["capability_samples"]
    if not samples:
        raise ValueError("eci-public returned no bootstrap samples")
    matrix = np.vstack([np.asarray(s, dtype=float) for s in samples])
    if matrix.shape[1] != len(names):
        raise ValueError("bootstrap sample width does not match the number of models")
    return names, matrix


def _read_frame(csv_bytes: bytes):
    import pandas as pd

    frame = pd.read_csv(io.BytesIO(csv_bytes))
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"eci_benchmarks.csv is missing columns {sorted(missing)}")
    return frame


def _from_matrix(names, matrix, *, seed: int, source_hash: str) -> EciBootstrap:
    draws = {name: matrix[:, i] for i, name in enumerate(names)}
    return EciBootstrap(draws, n_samples=matrix.shape[0], seed=seed, source_hash=source_hash)


def _load_cache(path: Path, source_hash: str, n_samples: int, seed: int) -> Optional[EciBootstrap]:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            if (str(data["source_hash"]) != source_hash
                    or int(data["seed"]) != seed
                    or int(data["requested_samples"]) != n_samples):
                return None
            names = [str(x) for x in data["model_names"]]
            matrix = np.asarray(data["matrix"], dtype=float)
    except Exception as error:
        logger.warning("Ignoring unreadable ECI bootstrap cache: %s", error)
        return None
    return _from_matrix(names, matrix, seed=seed, source_hash=source_hash)


def _save_cache(path: Path, names, matrix, source_hash: str, n_samples: int, seed: int) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, model_names=np.array(names, dtype=object), matrix=matrix,
                            source_hash=source_hash, requested_samples=n_samples, seed=seed)
    except Exception as error:
        logger.warning("Could not write the ECI bootstrap cache: %s", error)


def build_eci_bootstrap(csv_bytes: bytes, *, n_samples: int = BOOTSTRAP_SAMPLES,
                        seed: int = BOOTSTRAP_SEED, cache_dir=None) -> Optional[EciBootstrap]:
    """Fit the ECI with bootstrap on eci_benchmarks.csv and return the paired
    draws, or None on any failure. `cache_dir` enables the on-disk cache."""
    try:
        source_hash = hashlib.sha256(csv_bytes).hexdigest()
        cache_path = Path(cache_dir) / CACHE_FILENAME if cache_dir else None
        if cache_path is not None:
            cached = _load_cache(cache_path, source_hash, n_samples, seed)
            if cached is not None:
                logger.info("ECI bootstrap loaded from cache (%d models, %d resamples)",
                            len(cached.draws), cached.n_samples)
                return cached

        names, matrix = _fit_capability_draws(_read_frame(csv_bytes), n_samples, seed)
        if cache_path is not None:
            _save_cache(cache_path, names, matrix, source_hash, n_samples, seed)
        logger.info("ECI bootstrap fitted (%d models, %d resamples)", len(names), matrix.shape[0])
        return _from_matrix(names, matrix, seed=seed, source_hash=source_hash)
    except Exception as error:
        logger.warning("ECI bootstrap unavailable: %s", error)
        return None
