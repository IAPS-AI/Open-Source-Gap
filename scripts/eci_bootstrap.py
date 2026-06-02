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
        # Guard against non-finite draws: NaN comparisons silently read as
        # "not exceeds" and would bias P(a>b) downward. eci-public drops failed
        # resamples rather than NaN-filling, so this normally never triggers.
        x, y = da[:n], db[:n]
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            return None
        return float((x[mask] > y[mask]).mean())


def _fetch_csv_text(url: str, timeout: int = 120) -> str:
    import requests
    resp = requests.get(url, headers={"User-Agent": EPOCH_UA}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _read_validated_df(text: str):
    import pandas as pd
    df = pd.read_csv(StringIO(text))
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"eci_benchmarks.csv missing columns: {sorted(missing)}")
    return df


def _fit_capability_draws(df, n_samples: int, seed: int,
                          use_analytical_jacobian: bool):
    """Run eci-public's bootstrap fit; return (model_names, matrix[B, n_models])."""
    from eci import fit_eci_model
    _, _, bootstrap = fit_eci_model(
        df,
        bootstrap_samples=n_samples,
        bootstrap_seed=seed,
        use_analytical_jacobian=use_analytical_jacobian,
        return_bootstrap_samples=True,
    )
    model_names = list(bootstrap["model_names"])
    samples = bootstrap["capability_samples"]
    if not samples:
        raise ValueError("eci-public returned no bootstrap samples")
    matrix = np.vstack([np.asarray(s, dtype=float) for s in samples])
    if matrix.shape[1] != len(model_names):
        raise ValueError("bootstrap sample width != number of model names")
    return model_names, matrix


def _draws_from_matrix(names, matrix) -> dict[str, np.ndarray]:
    return {name: matrix[:, i] for i, name in enumerate(names)}


def _load_cache(cache_path: Path, source_hash: str, n_samples: int,
                seed: int) -> Optional["EciBootstrap"]:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=True) as data:
            if (str(data["source_hash"]) != source_hash
                    or int(data["seed"]) != seed
                    or int(data["requested_samples"]) != n_samples):
                return None
            names = [str(x) for x in data["model_names"]]
            matrix = np.asarray(data["matrix"], dtype=float)
    except Exception as e:
        logger.warning("Ignoring unreadable ECI bootstrap cache: %s", e)
        return None
    return EciBootstrap(_draws_from_matrix(names, matrix),
                        n_samples=matrix.shape[0], seed=seed,
                        source_hash=source_hash)


def _save_cache(cache_path: Path, names, matrix, source_hash: str,
                n_samples: int, seed: int) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            model_names=np.array(names, dtype=object),
            matrix=matrix,
            source_hash=source_hash,
            requested_samples=n_samples,
            seed=seed,
        )
    except Exception as e:
        logger.warning("Could not write ECI bootstrap cache: %s", e)


def build_eci_bootstrap(url: str = ECI_BENCHMARKS_URL,
                        n_samples: int = BOOTSTRAP_SAMPLES,
                        seed: int = BOOTSTRAP_SEED,
                        use_analytical_jacobian: bool = True,
                        cache_dir=None) -> Optional["EciBootstrap"]:
    """Fetch eci_benchmarks.csv, fit with bootstrap, and return paired
    capability draws keyed by ``Model``. Returns None on ANY failure so the
    daily build never breaks. ``cache_dir`` (e.g. "data") caches the matrix
    keyed by a hash of the source CSV; pass None to disable caching."""
    try:
        text = _fetch_csv_text(url)
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        cache_path = Path(cache_dir) / CACHE_FILENAME if cache_dir else None
        if cache_path is not None:
            cached = _load_cache(cache_path, source_hash, n_samples, seed)
            if cached is not None:
                logger.info("Loaded ECI bootstrap from cache (%d models, B=%d)",
                            len(cached.draws), cached.n_samples)
                return cached

        df = _read_validated_df(text)
        names, matrix = _fit_capability_draws(df, n_samples, seed,
                                              use_analytical_jacobian)
        boot = EciBootstrap(_draws_from_matrix(names, matrix),
                            n_samples=matrix.shape[0], seed=seed,
                            source_hash=source_hash)
        if cache_path is not None:
            _save_cache(cache_path, names, matrix, source_hash, n_samples, seed)
        logger.info("Fitted ECI bootstrap: %d models, B=%d",
                    len(boot.draws), boot.n_samples)
        return boot
    except Exception as e:
        logger.warning(
            "ECI bootstrap unavailable, falling back to analytical criterion: %s", e)
        return None
