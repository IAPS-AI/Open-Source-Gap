"""
METR Time Horizon Benchmark Data Fetcher

Fetches and merges METR benchmark results from v1.0 and v1.1 YAML files.
v1.1 data takes precedence over v1.0 for models that appear in both.

Data sources:
- https://metr.org/assets/benchmark_results_1_0.yaml
- https://metr.org/assets/benchmark_results_1_1.yaml

Blog posts:
- https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/
- https://metr.org/blog/2026-1-29-time-horizon-1-1/
"""

import logging
import re
from datetime import date, datetime
from typing import Any, Optional

import requests
import yaml

logger = logging.getLogger(__name__)

METR_V1_0_URL = "https://metr.org/assets/benchmark_results_1_0.yaml"
METR_V1_1_URL = "https://metr.org/assets/benchmark_results_1_1.yaml"

# Organization mapping for known model prefixes
MODEL_ORG_MAP = {
    "claude": "Anthropic",
    "gpt": "OpenAI",
    "o1": "OpenAI",
    "o3": "OpenAI",
    "o4": "OpenAI",
    "deepseek": "DeepSeek",
    "llama": "Meta",
    "qwen": "Alibaba",
    "qwq": "Alibaba",
    "gemini": "Google DeepMind",
    "mistral": "Mistral AI",
    "command": "Cohere",
    "davinci": "OpenAI",
    "grok": "xAI",
    "kimi": "Moonshot",
}

# Chinese organizations (matching the main app's classification)
CHINA_ORGANIZATIONS = {
    "DeepSeek", "Alibaba", "Baichuan", "01.AI", "Moonshot",
    "Tsinghua", "ByteDance", "Tencent", "Huawei", "SenseTime",
    "iFlytek", "Zhipu AI",
}

US_ORGANIZATIONS = {
    "OpenAI", "Anthropic", "Google DeepMind", "Meta",
    "Microsoft", "xAI", "Cohere",
}

# Open-weight model patterns
OPEN_MODEL_PATTERNS = [
    "deepseek", "llama", "qwen", "qwq", "mistral_large",
    "command_r", "gpt_oss", "gpt-oss", "kimi_k2",
]


def _normalize_model_id(model_id: str) -> str:
    """
    Normalize model ID for deduplication.

    v1.1 appends '_inspect' to some model IDs and may use slightly different
    naming. This strips the suffix and normalizes for comparison.
    """
    normalized = model_id.strip()
    # Remove _inspect suffix (v1.1 infrastructure marker)
    normalized = re.sub(r"_inspect$", "", normalized)
    # Normalize specific known equivalents
    # v1.1 uses gpt_5_2025_08_07 for gpt_5
    normalized = re.sub(r"gpt_5_\d{4}_\d{2}_\d{2}", "gpt_5", normalized)
    # v1.1 uses o1 for o1_elicited
    if normalized == "o1":
        normalized = "o1_elicited"
    # v1.0 uses claude_3_5_sonnet, v1.1 uses claude_3_5_sonnet_20240620
    if normalized == "claude_3_5_sonnet_20240620":
        normalized = "claude_3_5_sonnet"
    return normalized


def _build_display_name(model_id: str) -> str:
    """Build a human-readable display name from model ID."""
    # Strip _inspect suffix for display
    clean = re.sub(r"_inspect$", "", model_id)
    name = clean.replace("_", " ").replace("-", " ")
    # Title case, then fix known names
    name = name.title()
    replacements = {
        "Gpt": "GPT", "Gpt2": "GPT-2", "Deepseek": "DeepSeek",
        "Qwen": "Qwen", "Qwq": "QwQ", "Llama": "Llama",
        "4O": "4o", "4 O": "4o", "Turbo": "Turbo",
        "Davinci": "Davinci", "Grok": "Grok", "Kimi": "Kimi",
        "Gemini": "Gemini", "Oss": "OSS",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.strip()


def _infer_organization(model_id: str) -> str:
    """Infer organization from model ID."""
    model_lower = model_id.lower().replace("-", "_")
    for prefix, org in MODEL_ORG_MAP.items():
        if model_lower.startswith(prefix):
            return org
    return "Unknown"


def _is_open_model(model_id: str) -> bool:
    """Determine if a model is open-weight."""
    model_lower = model_id.lower().replace("-", "_")
    for pattern in OPEN_MODEL_PATTERNS:
        if pattern in model_lower:
            return True
    return False


def _parse_date(val) -> Optional[datetime]:
    """Parse a date from various formats returned by YAML."""
    if isinstance(val, datetime):
        return val
    if isinstance(val, date):
        return datetime(val.year, val.month, val.day)
    if isinstance(val, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                return datetime.strptime(val.split("+")[0].replace("Z", ""), fmt.replace("%z", ""))
            except ValueError:
                continue
    return None


def _parse_model_entry(model_id: str, model_data: dict, source_version: str) -> Optional[dict]:
    """Parse a single model entry from METR YAML data."""
    if not isinstance(model_data, dict):
        return None

    # Extract release date
    release_date = model_data.get("release_date")
    dt = _parse_date(release_date)
    if dt is None:
        logger.warning(f"Cannot parse date for {model_id}: {release_date}")
        return None

    # Metrics are nested under 'metrics' key
    metrics = model_data.get("metrics", {})
    if not isinstance(metrics, dict):
        return None

    # Extract average_score
    avg_score_data = metrics.get("average_score", {})
    if isinstance(avg_score_data, dict):
        average_score = avg_score_data.get("estimate")
    elif isinstance(avg_score_data, (int, float)):
        average_score = float(avg_score_data)
    else:
        average_score = None

    if average_score is None:
        logger.debug(f"Skipping {model_id}: no average_score")
        return None

    # Extract p50 horizon length
    p50_data = metrics.get("p50_horizon_length", {})
    p50_horizon = None
    p50_ci_low = None
    p50_ci_high = None
    if isinstance(p50_data, dict):
        p50_horizon = p50_data.get("estimate")
        p50_ci_low = p50_data.get("ci_low")
        p50_ci_high = p50_data.get("ci_high")
    elif isinstance(p50_data, (int, float)):
        p50_horizon = float(p50_data)

    # Extract p80 horizon length
    p80_data = metrics.get("p80_horizon_length", {})
    p80_horizon = None
    if isinstance(p80_data, dict):
        p80_horizon = p80_data.get("estimate")
    elif isinstance(p80_data, (int, float)):
        p80_horizon = float(p80_data)

    is_sota = metrics.get("is_sota", False)
    org = _infer_organization(model_id)

    return {
        "model": model_id,
        "normalized_id": _normalize_model_id(model_id),
        "display_name": _build_display_name(model_id),
        "date": dt.strftime("%Y-%m-%dT00:00:00"),
        "organization": org,
        "is_open": _is_open_model(model_id),
        "is_china": org in CHINA_ORGANIZATIONS,
        "is_us": org in US_ORGANIZATIONS,
        "average_score": float(average_score),
        "p50_horizon_minutes": float(p50_horizon) if p50_horizon is not None else None,
        "p50_ci_low": float(p50_ci_low) if p50_ci_low is not None else None,
        "p50_ci_high": float(p50_ci_high) if p50_ci_high is not None else None,
        "p80_horizon_minutes": float(p80_horizon) if p80_horizon is not None else None,
        "is_sota": bool(is_sota),
        "source_version": source_version,
        "benchmark_name": model_data.get("benchmark_name", f"METR-Horizon-v{source_version}"),
    }


def fetch_metr_yaml(url: str) -> Optional[dict]:
    """Fetch and parse a METR YAML file."""
    logger.info(f"Fetching METR data from {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = yaml.safe_load(response.text)
        return data
    except requests.RequestException as e:
        logger.error(f"Failed to fetch METR data from {url}: {e}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse METR YAML from {url}: {e}")
        return None


def fetch_metr_data() -> Optional[dict]:
    """
    Fetch and merge METR benchmark data from v1.0 and v1.1.

    v1.1 takes precedence over v1.0 for duplicate models (matched by
    normalized model ID).

    Returns:
        Dictionary with merged model data and metadata, or None on failure.
    """
    v1_0_raw = fetch_metr_yaml(METR_V1_0_URL)
    v1_1_raw = fetch_metr_yaml(METR_V1_1_URL)

    if v1_0_raw is None and v1_1_raw is None:
        logger.error("Failed to fetch any METR data")
        return None

    # Parse models from each version
    v1_0_models = {}  # keyed by normalized_id
    v1_1_models = {}

    if v1_0_raw:
        results = v1_0_raw.get("results", {})
        for model_id, model_data in results.items():
            parsed = _parse_model_entry(model_id, model_data, "1.0")
            if parsed:
                v1_0_models[parsed["normalized_id"]] = parsed

    if v1_1_raw:
        results = v1_1_raw.get("results", {})
        for model_id, model_data in results.items():
            parsed = _parse_model_entry(model_id, model_data, "1.1")
            if parsed:
                v1_1_models[parsed["normalized_id"]] = parsed

    # Merge: v1.1 takes precedence over v1.0
    merged = {**v1_0_models, **v1_1_models}

    v1_0_only = set(v1_0_models.keys()) - set(v1_1_models.keys())
    v1_1_only = set(v1_1_models.keys()) - set(v1_0_models.keys())
    overlap = set(v1_0_models.keys()) & set(v1_1_models.keys())

    logger.info(
        f"METR merge: {len(v1_0_models)} from v1.0, {len(v1_1_models)} from v1.1, "
        f"{len(overlap)} overlapping (v1.1 wins), {len(merged)} total"
    )

    # Extract doubling time metadata (prefer v1.1)
    meta_source = v1_1_raw or v1_0_raw
    doubling_time_days = None
    doubling_time_ci = None
    post_2023_doubling = None

    if meta_source:
        dt_data = meta_source.get("doubling_time_in_days", {})
        if isinstance(dt_data, dict):
            # All-time
            all_time = dt_data.get("all_time", dt_data.get("all_time_stitched", {}))
            if isinstance(all_time, dict):
                doubling_time_days = all_time.get("point_estimate")
                ci_low = all_time.get("ci_low")
                ci_high = all_time.get("ci_high")
                if ci_low is not None and ci_high is not None:
                    doubling_time_ci = [float(ci_low), float(ci_high)]
            # Post-2023
            from_2023 = dt_data.get("from_2023_on", dt_data.get("from_2023", {}))
            if isinstance(from_2023, dict):
                post_2023_doubling = from_2023.get("point_estimate")

    models_list = sorted(merged.values(), key=lambda m: m["date"])

    # Remove the normalized_id from output (internal use only)
    for m in models_list:
        m.pop("normalized_id", None)

    return {
        "models": models_list,
        "metadata": {
            "id": "metr_time_horizon",
            "name": "METR Time Horizon",
            "description": (
                "Measures how long tasks (in human-expert minutes) AI agents can "
                "autonomously complete. The p50 horizon is the task duration where "
                "the model achieves 50% success probability."
            ),
            "unit": "p50 Horizon (minutes)",
            "score_field": "p50_horizon_minutes",
            "secondary_score_field": "average_score",
            "threshold": 0.5,  # Ratio: match if within 50% of reference horizon
            "threshold_type": "ratio",  # Use multiplicative matching, not additive
            "scale": 1,
            "source": "METR",
            "source_url": "https://metr.org/blog/2026-1-29-time-horizon-1-1/",
            "doubling_time_days": float(doubling_time_days) if doubling_time_days else None,
            "doubling_time_ci": doubling_time_ci,
            "post_2023_doubling_time_days": float(post_2023_doubling) if post_2023_doubling else None,
            "versions_merged": ["1.0", "1.1"],
        },
        "v1_0_count": len(v1_0_models),
        "v1_1_count": len(v1_1_models),
        "overlap_count": len(overlap),
    }


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)
    data = fetch_metr_data()
    if data:
        print(f"\nFetched {len(data['models'])} models")
        print(f"  v1.0 only: {data['v1_0_count'] - data['overlap_count']}")
        print(f"  v1.1 only: {data['v1_1_count'] - data['overlap_count']}")
        print(f"  overlap (v1.1 used): {data['overlap_count']}")
        print(f"\nMetadata: {json.dumps(data['metadata'], indent=2)}")
        print(f"\nModels:")
        for m in data["models"]:
            horizon = m.get("p50_horizon_minutes")
            horizon_str = f"{horizon:.1f} min" if horizon else "N/A"
            print(
                f"  {m['display_name']:40s} "
                f"score={m['average_score']:.3f}  "
                f"p50={horizon_str:>12s}  "
                f"{'Open' if m['is_open'] else 'Closed':6s}  "
                f"{m['organization']:20s}  "
                f"v{m['source_version']}  "
                f"{m['date'][:10]}"
            )
    else:
        print("Failed to fetch data")
