"""
CSV-based Benchmark Data Fetcher

Fetches benchmark data directly from Epoch AI's public CSV exports.
This replaces the Airtable-based fetcher and doesn't require any credentials.

Data source: https://epoch.ai/data/benchmark_data.zip
Updated daily by Epoch AI.
"""

import io
import logging
import os
import zipfile
from datetime import datetime
from typing import Optional
import pandas as pd
import requests

from accessibility import (
    ACCESSIBILITY_OVERRIDES,
    AccessibilitySourceError,
    classify_accessibility,
    report_unrecognized_labels,
)

logger = logging.getLogger(__name__)

# URL for Epoch AI's benchmark data export
BENCHMARK_DATA_URL = "https://epoch.ai/data/benchmark_data.zip"

# Per-model-version metadata shipped in the same zip (Model version slug ->
# accessibility, display name). Replaced epoch_capabilities_index.csv, which
# Epoch moved out of the zip root around 2026-09-05.
MODEL_METADATA_FILE = "model_metadata.csv"
MODEL_METADATA_COLUMNS = ("model_version", "accessibility", "display_name")

# Minimum share of a benchmark's scored rows that must carry a usable Epoch
# accessibility label. Today's range is ~94-100%; a drop below this means the
# label source moved or broke, so the build fails rather than publishing a
# frontier built from overrides alone.
MIN_EPOCH_LABEL_COVERAGE = 0.85

# Mapping of benchmark IDs to CSV filenames and configuration
BENCHMARK_CSV_CONFIG = {
    "gpqa_diamond": {
        "filename": "gpqa_diamond.csv",
        "name": "GPQA Diamond",
        "description": "Graduate-level science questions (Diamond subset)",
        "unit": "Accuracy",
        "scale": 100,  # Scores are 0-1, multiply by 100 for percentage
        "threshold": 1.0,
    },
    "math_level_5": {
        "filename": "math_level_5.csv",
        "name": "MATH Level 5",
        "description": "Competition mathematics problems (hardest level)",
        "unit": "Accuracy",
        "scale": 100,
        "threshold": 1.0,
    },
    "swe_bench_verified": {
        "filename": "swe_bench_verified.csv",
        "name": "SWE-Bench Verified",
        "description": "Software engineering bug fixing benchmark",
        "unit": "Resolve Rate",
        "scale": 100,
        "threshold": 1.0,
    },
    "frontiermath_public": {
        "filename": "frontiermath.csv",
        "name": "FrontierMath",
        "description": "Frontier-level mathematics problems",
        "unit": "Accuracy",
        "scale": 100,
        "threshold": 0.5,
    },
    "simpleqa_verified": {
        "filename": "simpleqa_verified.csv",
        "name": "SimpleQA Verified",
        "description": "Simple factual question answering",
        "unit": "Accuracy",
        "scale": 100,
        "threshold": 1.0,
    },
    "chess_puzzles": {
        "filename": "chess_puzzles.csv",
        "name": "Chess Puzzles",
        "description": "Chess tactical puzzles",
        "unit": "Accuracy",
        "scale": 100,
        "threshold": 1.0,
    },
    "otis_mock_aime": {
        "filename": "otis_mock_aime_2024_2025.csv",
        "name": "OTIS Mock AIME",
        "description": "Mock AIME competition problems",
        "unit": "Score",
        "scale": 100,
        "threshold": 1.0,
    },
}

# Organizations known to be Chinese
CHINA_ORGS = {
    '01.ai', 'alibaba', 'baidu', 'bytedance', 'deepseek', 'iflytek',
    'huawei', 'tencent', 'zhipu', 'z.ai', 'shanghai ai', 'baichuan',
    'thudm', 'moonshot', 'sensetime', 'idea research', 'modelbest',
    'qwen', 'tsinghua', 'peking university', 'chinese academy'
}

# There are deliberately no org/name heuristics: labs ship both open and
# closed models (Meta's API-only Muse line was counted as open by an
# "org == meta" rule), so a model without an Epoch label or a vendored
# override is excluded (audit finding F1,
# docs/audits/2026-07-01-calculation-logic-audit.md).


class CSVBenchmarkFetcher:
    """Fetches benchmark data from Epoch AI's CSV exports."""

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = cache_dir or "/tmp/epoch_benchmark_cache"
        self._zip_data = None
        self._csv_cache = {}
        self._accessibility_by_version: Optional[dict] = None
        self._display_name_by_version: Optional[dict] = None

    def _download_zip(self) -> bytes:
        """Download the benchmark data zip file."""
        if self._zip_data is not None:
            return self._zip_data

        logger.info(f"Downloading benchmark data from {BENCHMARK_DATA_URL}...")
        # epoch.ai blocks the default urllib User-Agent with 403, so use
        # requests with a browser-like UA.
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(BENCHMARK_DATA_URL, headers=headers, timeout=60)
        response.raise_for_status()
        self._zip_data = response.content
        logger.info(f"  Downloaded {len(self._zip_data) / 1024:.1f} KB")
        return self._zip_data

    def _read_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Read a CSV file from the zip archive."""
        if filename in self._csv_cache:
            return self._csv_cache[filename]

        zip_data = self._download_zip()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            # List available files for debugging
            available_files = zf.namelist()

            if filename not in available_files:
                logger.warning(f"File {filename} not found in zip. Available: {available_files[:10]}...")
                return None

            with zf.open(filename) as f:
                df = pd.read_csv(f)
                self._csv_cache[filename] = df
                return df

    def _is_china_org(self, org: str) -> bool:
        """Check if organization is Chinese."""
        if not org or pd.isna(org):
            return False
        org_lower = org.lower()
        return any(china_org in org_lower for china_org in CHINA_ORGS)

    def _load_model_metadata(self) -> None:
        """Build Model version -> accessibility / display name maps from
        model_metadata.csv. The benchmark CSVs carry no accessibility column,
        so this join is the primary classification source. Raises
        AccessibilitySourceError if the file or its columns are missing."""
        meta = self._read_csv(MODEL_METADATA_FILE)
        missing = ([] if meta is None else
                   [c for c in MODEL_METADATA_COLUMNS if c not in meta.columns])
        if meta is None or missing:
            raise AccessibilitySourceError(
                f"{MODEL_METADATA_FILE} unavailable in {BENCHMARK_DATA_URL}"
                + (f" (missing columns {missing})" if missing else "")
                + "; refusing to classify open/closed without Epoch labels")

        meta = meta[meta["model_version"].notna()].copy()
        meta["model_version"] = meta["model_version"].astype(str).str.strip()
        meta = meta.drop_duplicates(subset=["model_version"], keep="first")
        labelled = meta[meta["accessibility"].notna()]
        named = meta[meta["display_name"].notna()]
        self._accessibility_by_version = dict(
            zip(labelled["model_version"], labelled["accessibility"]))
        self._display_name_by_version = dict(
            zip(named["model_version"], named["display_name"].astype(str).str.strip()))
        report_unrecognized_labels(self._accessibility_by_version.values(),
                                   MODEL_METADATA_FILE)

    def _get_accessibility_map(self) -> dict:
        if self._accessibility_by_version is None:
            self._load_model_metadata()
        return self._accessibility_by_version

    def _get_display_name_map(self) -> dict:
        if self._display_name_by_version is None:
            self._load_model_metadata()
        return self._display_name_by_version

    def _classify_model(self, row: pd.Series) -> Optional[bool]:
        """True = open weights, False = closed, None = unclassifiable
        (callers must exclude None rows rather than guess).

        Precedence: the CSV's own accessibility column (if ever present) >
        Epoch model metadata > vendored overrides > None."""
        label = classify_accessibility(row.get('Model accessibility'))
        if label is not None:
            return label

        name = str(row.get('Model version', row.get('Model name', ''))).strip()
        label = classify_accessibility(self._get_accessibility_map().get(name))
        if label is not None:
            return label
        return classify_accessibility(ACCESSIBILITY_OVERRIDES.get(name))

    def get_available_benchmarks(self) -> list:
        """Return list of available benchmarks."""
        available = []
        for bench_id, config in BENCHMARK_CSV_CONFIG.items():
            df = self._read_csv(config["filename"])
            if df is not None and len(df) > 0:
                available.append({
                    "id": bench_id,
                    "name": config["name"],
                    "description": config["description"],
                    "unit": config["unit"],
                    "model_count": len(df)
                })
        return available

    def fetch_benchmark_data(self, benchmark_id: str) -> Optional[dict]:
        """
        Fetch and transform data for a specific benchmark.

        Returns data in the same format as the Airtable-based fetcher.
        """
        if benchmark_id not in BENCHMARK_CSV_CONFIG:
            logger.warning(f"Unknown benchmark: {benchmark_id}")
            return None

        config = BENCHMARK_CSV_CONFIG[benchmark_id]
        df = self._read_csv(config["filename"])

        if df is None or len(df) == 0:
            logger.warning(f"No data found for {benchmark_id}")
            return None

        logger.info(f"Processing {config['name']} ({len(df)} rows)...")

        accessibility_map = self._get_accessibility_map()
        display_names = self._get_display_name_map()

        models = []
        excluded_unclassifiable = []
        n_candidates = 0
        n_epoch_labelled = 0
        for _, row in df.iterrows():
            # Get score - check various column names
            score = None
            for score_col in ['mean_score', 'Best score (across scorers)', 'ECI Score']:
                if score_col in row and pd.notna(row[score_col]):
                    score = float(row[score_col])
                    break

            if score is None:
                continue

            # Scale score if needed (0-1 to percentage)
            if config["scale"] != 1 and score <= 1.0:
                score = score * config["scale"]

            # Get release date
            date_str = row.get('Release date')
            if pd.isna(date_str):
                continue

            try:
                # Parse date and convert to ISO format
                date = pd.to_datetime(date_str)
                date_iso = date.strftime("%Y-%m-%dT%H:%M:%S")
            except Exception:
                continue

            # Get model name
            model_name = row.get('Model version', row.get('Model name', 'Unknown'))
            if pd.isna(model_name):
                continue

            # Get organization
            org = row.get('Organization', 'Unknown')
            if pd.isna(org):
                org = 'Unknown'

            # Get stderr if available
            stderr = row.get('stderr', 0)
            if pd.isna(stderr):
                stderr = 0
            else:
                stderr = float(stderr)
                if config["scale"] != 1 and stderr <= 1.0:
                    stderr = stderr * config["scale"]

            slug = str(model_name).strip()
            n_candidates += 1
            if classify_accessibility(accessibility_map.get(slug)) is not None:
                n_epoch_labelled += 1

            # Determine if open/closed and China/US. Unclassifiable models
            # are excluded — guessing poisons the frontier (audit F1).
            is_open = self._classify_model(row)
            if is_open is None:
                excluded_unclassifiable.append(str(model_name))
                continue
            is_china = self._is_china_org(org)

            # Get country
            country = row.get('Country', '')
            if pd.isna(country):
                country = ''

            models.append({
                "model": str(model_name),
                # Epoch's human-readable name ("Gemini 4 Argon (high)");
                # older rows ship it blank, so fall back to the slug.
                "display_name": display_names.get(slug, str(model_name)),
                "score": round(score, 2),
                "score_std": round(stderr, 2),
                "date": date_iso,
                "organization": str(org),
                "is_open": is_open,
                "is_china": is_china,
                "country": str(country),
            })

        if n_candidates:
            coverage = n_epoch_labelled / n_candidates
            logger.info(f"  Epoch accessibility labels cover {coverage:.1%} "
                        f"of {n_candidates} scored rows")
            if coverage < MIN_EPOCH_LABEL_COVERAGE:
                raise AccessibilitySourceError(
                    f"{benchmark_id}: only {coverage:.1%} of {n_candidates} scored "
                    f"rows have an Epoch accessibility label in {MODEL_METADATA_FILE} "
                    f"(minimum {MIN_EPOCH_LABEL_COVERAGE:.0%}); the label source "
                    "has likely moved or changed format")

        # Sort by date
        models.sort(key=lambda x: x['date'])

        # Log summary
        open_count = sum(1 for m in models if m['is_open'])
        closed_count = len(models) - open_count
        logger.info(f"  Found {len(models)} models ({open_count} open, {closed_count} closed)")
        if excluded_unclassifiable:
            logger.info(f"  Excluded {len(excluded_unclassifiable)} unclassifiable: "
                        f"{excluded_unclassifiable[:8]}")

        return {
            "models": models,
            "metadata": {
                "id": benchmark_id,
                "name": config["name"],
                "description": config["description"],
                "unit": config["unit"],
                "threshold": config["threshold"],
                "scale": config["scale"],
                "source": "Epoch AI CSV Export",
            }
        }


def main():
    """Test the CSV benchmark fetcher."""
    logging.basicConfig(level=logging.INFO)

    fetcher = CSVBenchmarkFetcher()

    print("\n" + "=" * 60)
    print("AVAILABLE BENCHMARKS")
    print("=" * 60)

    benchmarks = fetcher.get_available_benchmarks()
    for b in benchmarks:
        print(f"  {b['name']} ({b['id']}): {b['model_count']} models")

    print("\n" + "=" * 60)
    print("TEST: GPQA DIAMOND DATA")
    print("=" * 60)

    data = fetcher.fetch_benchmark_data("gpqa_diamond")
    if data:
        print(f"\nMetadata: {data['metadata']}")
        print(f"\nSample models:")
        for m in data['models'][:5]:
            print(f"  {m['display_name']}: {m['score']:.1f}% ({m['organization']}, "
                  f"{'Open' if m['is_open'] else 'Closed'})")


if __name__ == "__main__":
    main()
