"""The static site: the page, the script, the published data.json, and the
deploy workflow have to agree with each other, and nothing else checks that."""
import json
import re
from datetime import date, timedelta

import pytest

from conftest import ROOT

INDEX = (ROOT / "index.html").read_text(encoding="utf-8")
SCRIPT = (ROOT / "static" / "script.js").read_text(encoding="utf-8")
STYLE = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
DATA = json.loads((ROOT / "data.json").read_text(encoding="utf-8"))
DEPLOY = (ROOT / ".github" / "workflows" / "deploy_pages.yml").read_text(encoding="utf-8")

COMPARISONS = sorted(DATA["comparisons"].items())
each_comparison = pytest.mark.parametrize("name, comparison", COMPARISONS, ids=[name for name, _ in COMPARISONS])


def day(text):
    return date.fromisoformat(text)


# ------------------------------------------------------------------- page

def test_every_element_the_script_looks_up_exists_on_the_page():
    wanted = set(re.findall(r"\$\('([^']+)'\)", SCRIPT))
    assert wanted, "the script should look elements up by id"
    present = set(re.findall(r'\bid="([^"]+)"', INDEX))
    assert wanted <= present, f"missing ids: {sorted(wanted - present)}"


def test_the_page_loads_only_its_own_script_and_a_versioned_stylesheet():
    (script,) = re.findall(r'<script[^>]*\bsrc="([^"]+)"', INDEX)       # nothing third-party
    (sheet,) = re.findall(r'<link[^>]*rel="stylesheet"[^>]*href="(static/[^"]+)"', INDEX)
    # Pages serves the branch as committed, so this number is what busts caches.
    assert re.fullmatch(r"static/script\.js\?v=\d+", script)
    assert re.fullmatch(r"static/style\.css\?v=\d+", sheet)
    assert script.split("=")[1] == sheet.split("=")[1]


def test_deploy_cache_busting_still_matches_the_asset_urls():
    """deploy_pages.yml stamps both asset URLs with the commit and then greps
    for the result, so a renamed asset would fail the deploy."""
    assert 's#static/script\\.js(\\?v=[^\\"]*)?\\"#' in DEPLOY
    assert 's#static/style\\.css(\\?v=[^\\"]*)?\\"#' in DEPLOY
    stamped = re.sub(r'static/script\.js(\?v=[^"]*)?"', 'static/script.js?v=abc12345"', INDEX)
    stamped = re.sub(r'static/style\.css(\?v=[^"]*)?"', 'static/style.css?v=abc12345"', stamped)
    assert 'static/script.js?v=abc12345"' in stamped
    assert 'static/style.css?v=abc12345"' in stamped


def test_chart_height_is_reserved_with_the_sums_the_script_uses():
    """The stylesheet holds the chart's space before it is drawn. If these
    drift apart the page jumps when the chart appears."""
    panels = "clamp(220px, 34cqw, 360px) + clamp(150px, 22cqw, 230px)"
    assert f"calc(96px + {panels})" in STYLE                 # 24 + 46 + 26
    assert f"calc(118px + {panels})" in STYLE                # 24 + 68 + 26 on a narrow chart
    assert "@container (max-width: 519.98px)" in STYLE
    assert "const narrow = width < 520;" in SCRIPT
    assert "top: 24, bottom: 26 };" in SCRIPT
    assert "const between = narrow ? 68 : 46;" in SCRIPT
    assert "clamp(Math.round(width * 0.34), 220, 360)" in SCRIPT
    assert "clamp(Math.round(width * 0.22), 150, 230)" in SCRIPT


def rule(selector):
    """The declarations of the one rule in the stylesheet with exactly this selector."""
    (body,) = re.findall(re.escape(selector) + r"\s*\{([^}]*)\}", STYLE)
    return dict(re.findall(r"([a-z0-9-]+)\s*:\s*([^;]+);", body))


def test_both_themes_define_the_same_colour_tokens():
    light = rule("\n:root")
    dark_by_system = rule(':root:not([data-theme="light"])')
    dark_by_choice = rule(':root[data-theme="dark"]')
    # The dark values are written twice (system setting, reader's choice) and must not drift.
    assert dark_by_system == dark_by_choice
    assert set(dark_by_choice) == set(light) - {"--font"}
    assert (light["color-scheme"], dark_by_choice["color-scheme"]) == ("light", "dark")
    used = set(re.findall(r"var\((--[a-z0-9-]+)\)", STYLE))
    assert used <= set(light), f"undefined tokens: {sorted(used - set(light))}"
    # The China view's accent is set for light and for both routes to dark.
    china = [rule(selector)["--laggard"] for selector in (
        '\n:root[data-view="china_us"]',
        ':root:not([data-theme="light"])[data-view="china_us"]',
        ':root[data-theme="dark"][data-view="china_us"]')]
    assert china[1] == china[2] and china[0] != china[1]
    assert light["--laggard"] not in china and dark_by_choice["--laggard"] not in china


def test_the_theme_switch_is_wired_the_same_way_in_the_page_and_the_script():
    # The page's one inline script only restores a saved theme before first paint.
    (inline,) = re.findall(r"<script>(.*?)</script>", INDEX, re.S)
    assert "localStorage.getItem('theme')" in inline and "dataset.theme = saved" in inline
    assert "fetch(" not in inline and "http" not in inline
    assert INDEX.index("<script>") < INDEX.index('rel="stylesheet"')        # ahead of anything painted
    # The script stores under the same key, and only the two values the page accepts.
    assert "localStorage.setItem('theme', theme)" in SCRIPT and "localStorage.removeItem('theme')" in SCRIPT
    assert "saved === 'light' || saved === 'dark'" in inline
    # The button cannot work without the script, so it stays hidden until the script shows it.
    assert '<button type="button" class="theme-toggle" id="theme-toggle" hidden>' in INDEX
    assert "button.hidden = false;" in SCRIPT


def test_the_wordmark_is_plain_text():
    """A logo mark of two dots read as a light/dark switch, so the name stands alone."""
    (wordmark,) = re.findall(r'<a class="wordmark"[^>]*>(.*?)</a>', INDEX, re.S)
    assert wordmark.strip() == "Open Source Gap"
    assert "glyph" not in INDEX and "glyph" not in STYLE


def test_nothing_from_the_old_multi_benchmark_site_is_left():
    for leftover in ("plotly", "benchmark-select", "framing-toggle", "time_horizon"):
        assert leftover not in INDEX.lower()
        assert leftover not in SCRIPT.lower()


# ------------------------------------------------------------------ views

def test_the_page_script_and_data_agree_on_the_comparisons():
    in_data = set(DATA["comparisons"])
    in_script = set(re.findall(r"^        (\w+): \{$", SCRIPT[SCRIPT.index("const WORDS"):SCRIPT.index("const HASHES")], re.M))
    assert in_script == in_data
    assert set(re.findall(r'data-view-link="(\w+)"', INDEX)) == in_data      # one link each
    assert set(re.findall(r'class="notes" data-view="(\w+)"', INDEX)) == in_data  # one set of notes each
    hashes = dict(re.findall(r"'(#[a-z-]+)': '(\w+)'", SCRIPT))
    assert set(hashes.values()) == in_data
    assert set(re.findall(r'href="(#[a-z-]+)" data-view-link', INDEX)) == set(hashes)


def test_only_the_default_views_notes_show_before_the_script_runs():
    blocks = re.findall(r'<div class="notes" data-view="(\w+)"( hidden)?>', INDEX)
    assert blocks == [("open_closed", ""), ("china_us", " hidden")]
    assert 'data-view-link="open_closed" aria-current="true"' in INDEX


# ------------------------------------------------------------- data.json

def test_data_json_is_the_schema_the_script_reads():
    assert f"const SCHEMA = {DATA['schema']};" in SCRIPT
    for key in ("as_of", "days_per_month", "test", "comparisons"):
        assert key in DATA
    assert {"paired_bootstrap", "resamples"} <= set(DATA["test"])
    assert list(DATA["comparisons"]) == ["open_closed", "china_us"]


@each_comparison
def test_comparison_has_the_fields_the_script_reads(name, comparison):
    for key in ("headline", "today", "segments", "average", "leader_frontier", "laggard_frontier", "trend"):
        assert key in comparison
    for key in ("months", "eci_points", "window_start", "window_end", "window_months", "full_history"):
        assert key in comparison["headline"]
    for key in ("months", "eci_points", "matched"):
        assert key in comparison["today"]
    assert set(comparison["average"]) == {"start", "firm_from", "months"}


def test_only_china_vs_us_carries_a_trendline():
    assert DATA["comparisons"]["open_closed"]["trend"] is None
    trend = DATA["comparisons"]["china_us"]["trend"]
    assert set(trend) == {"since", "lag_months", "change_months_per_year", "on", "laggard_model", "level",
                          "leader_reached", "laggard", "leader"}
    for group in (trend["laggard"], trend["leader"]):
        assert set(group) == {"records", "eci_per_month", "r2", "line"}
        assert group["records"] >= 3 and len(group["line"]) == 2


def test_the_trendline_figures_follow_from_the_lines_the_page_draws():
    """Rebuild each fitted line from its two published end points and read the
    gap off them, as the chart does."""
    comparison = DATA["comparisons"]["china_us"]
    trend = comparison["trend"]

    def line(group):
        (d0, e0), (d1, e1) = group["line"]
        slope = (e1 - e0) / (day(d1) - day(d0)).days
        return slope, lambda when: e0 + slope * (when - day(d0)).days

    laggard_slope, laggard_at = line(trend["laggard"])
    leader_slope, leader_at = line(trend["leader"])
    on = day(trend["on"])
    # It is read where the laggards' line stands on the day of their latest record.
    latest = comparison["laggard_frontier"][-1]
    assert (trend["on"], trend["laggard_model"]) == (latest["released"], latest["model"])
    assert trend["level"] == pytest.approx(laggard_at(on), abs=0.05)
    assert leader_at(day(trend["leader_reached"])) == pytest.approx(trend["level"], abs=0.1)
    assert trend["lag_months"] == pytest.approx(
        (on - day(trend["leader_reached"])).days / DATA["days_per_month"], abs=0.05)
    assert trend["change_months_per_year"] == pytest.approx((1 - laggard_slope / leader_slope) * 12, abs=0.05)
    for group, slope in ((trend["laggard"], laggard_slope), (trend["leader"], leader_slope)):
        assert group["eci_per_month"] == pytest.approx(slope * DATA["days_per_month"], abs=0.02)
    # Every fitted record is a dot on the chart: a true record, released since the cut-off.
    for group, frontier in ((trend["laggard"], comparison["laggard_frontier"]),
                            (trend["leader"], comparison["leader_frontier"])):
        assert group["records"] == sum(1 for m in frontier if m["released"] >= trend["since"])


@each_comparison
def test_every_model_a_segment_names_is_in_the_frontier_lists(name, comparison):
    leaders = {m["model"] for m in comparison["leader_frontier"]}
    laggards = {m["model"] for m in comparison["laggard_frontier"]}
    assert len(leaders) == len(comparison["leader_frontier"])
    assert len(laggards) == len(comparison["laggard_frontier"])
    for s in comparison["segments"]:
        assert s["laggard_model"] in laggards
        assert s["leader_model"] in leaders and s["reference_model"] in leaders
        assert s["unmatched_model"] is None or s["unmatched_model"] in leaders


@each_comparison
def test_frontiers_and_segments_are_in_order(name, comparison):
    for frontier in (comparison["leader_frontier"], comparison["laggard_frontier"]):
        released = [m["released"] for m in frontier]
        scores = [m["eci"] for m in frontier]
        assert released == sorted(released)
        assert scores == sorted(scores) and len(set(scores)) == len(scores)   # each one a new record
    starts = [s["start"] for s in comparison["segments"]]
    assert starts == sorted(set(starts))
    assert starts[-1] <= DATA["as_of"]


def daily_months(comparison):
    """Months behind on every day of the series, from the segments alone, the
    way the page computes it."""
    released = {m["model"]: day(m["released"]) for m in comparison["leader_frontier"]}
    segments = comparison["segments"]
    out = {}
    for i, s in enumerate(segments):
        when = day(s["start"])
        end = day(segments[i + 1]["start"]) if i + 1 < len(segments) else day(DATA["as_of"]) + timedelta(days=1)
        while when < end:
            out[when] = (when - released[s["reference_model"]]).days / DATA["days_per_month"]
            when += timedelta(days=1)
    return out


@each_comparison
def test_published_figures_follow_from_the_published_series(name, comparison):
    months = daily_months(comparison)
    eci = {m["model"]: m["eci"] for m in comparison["leader_frontier"] + comparison["laggard_frontier"]}
    last = comparison["segments"][-1]
    today = comparison["today"]
    assert today["months"] == round(months[day(DATA["as_of"])], 1)
    assert today["eci_points"] == pytest.approx(eci[last["leader_model"]] - eci[last["laggard_model"]], abs=0.051)
    assert (today["laggard_model"], today["reference_model"], today["matched"]) == (
        last["laggard_model"], last["reference_model"], last["matched"])

    headline = comparison["headline"]
    first, end = day(headline["window_start"]), day(headline["window_end"])
    window = [months[first + timedelta(days=i)] for i in range((end - first).days + 1)]
    assert headline["days"] == len(window)
    assert headline["months"] == round(sum(window) / len(window), 1)


@each_comparison
def test_rolling_average_runs_daily_to_the_headline(name, comparison):
    average, headline = comparison["average"], comparison["headline"]
    start, end = day(average["start"]), day(headline["window_end"])
    values = average["months"]
    assert start + timedelta(days=len(values) - 1) == end       # one value a day, ending on the headline's day
    assert round(values[-1], 1) == headline["months"]
    assert average["firm_from"] is None or start <= day(average["firm_from"]) <= end

    months = daily_months(comparison)
    for offset in (0, len(values) // 3, len(values) - 1):       # spot-check against the series
        when = start + timedelta(days=offset)
        first = when.replace(year=when.year - 1)                # no 29 February among these days
        window = [months[first + timedelta(days=i)] for i in range((when - first).days + 1)]
        assert values[offset] == pytest.approx(sum(window) / len(window), abs=0.006)


@each_comparison
def test_matched_records_report_a_consistent_lag(name, comparison):
    laggards = {m["model"] for m in comparison["laggard_frontier"]}
    for m in comparison["leader_frontier"]:
        if m["matched_by"] is None:
            assert m["matched_on"] is None and m["lag_months"] is None
            continue
        assert m["matched_by"] in laggards
        assert m["matched_on"] >= m["released"]
        lag = (day(m["matched_on"]) - day(m["released"])).days / DATA["days_per_month"]
        assert m["lag_months"] == round(lag, 1)
