"""
Regression check that the served index.html exposes the structural anchors
the gap-over-time chart's JS expects.

Run with: pytest tests/test_index_html_structure.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def test_index_serves_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_historical_chart_card_present(client):
    """JS targets #historical-chart, #historical-tooltip, and
    #historical-chart-download. The card wrapper makes them discoverable as
    a unit; if any anchor goes missing the chart breaks silently in the
    browser, so guard them here."""
    html = client.get("/").get_data(as_text=True)
    for anchor in (
        'id="historical-chart-card"',
        'id="historical-chart-container"',
        'id="historical-chart"',
        'id="historical-tooltip"',
        'id="historical-chart-download"',
    ):
        assert anchor in html, f"Missing anchor: {anchor}"


def test_script_js_referenced(client):
    html = client.get("/").get_data(as_text=True)
    assert "static/script.js" in html


def test_legacy_inline_styles_removed(client):
    """The old container used inline styles for height; the new card sizes
    via CSS. This guards against regressions where someone reverts the
    structure and breaks the responsive viewBox sizing."""
    html = client.get("/").get_data(as_text=True)
    assert 'id="historical-chart" style=' not in html


THRESHOLD_ANCHORS = (
    'id="threshold-section"',
    'id="threshold-chart"',
    'id="threshold-note"',
)


def test_threshold_section_present(client):
    """renderThresholdChart targets #threshold-section (for graceful hiding
    when data.json lacks threshold_aggregate), #threshold-chart (Plotly
    mount) and #threshold-note (medians caption)."""
    html = client.get("/").get_data(as_text=True)
    for anchor in THRESHOLD_ANCHORS:
        assert anchor in html, f"Missing anchor: {anchor}"


def test_threshold_section_in_root_index():
    root_html = (Path(__file__).parent.parent / "index.html").read_text(
        encoding="utf-8"
    )
    for anchor in THRESHOLD_ANCHORS:
        assert anchor in root_html, f"Missing anchor in root index.html: {anchor}"


def test_threshold_chart_renderer_wired():
    """script.js must define renderThresholdChart, read threshold_aggregate,
    and invoke the renderer (init-time; the aggregate is benchmark- and
    framing-independent)."""
    js = (Path(__file__).parent.parent / "static" / "script.js").read_text(
        encoding="utf-8"
    )
    assert "function renderThresholdChart" in js
    assert "threshold_aggregate" in js
    assert "renderThresholdChart(" in js.replace(
        "function renderThresholdChart(", "", 1
    )


def test_root_static_index_matches_chart_structure():
    """GitHub Pages serves the root index.html (not the Flask template), so
    chart-structure changes must be mirrored there too. Drifting markup was
    the cause of an overlap bug where the gap chart bled into the next
    section in production."""
    root_html = (Path(__file__).parent.parent / "index.html").read_text(
        encoding="utf-8"
    )
    for anchor in (
        'id="historical-chart-card"',
        'id="historical-chart-container"',
        'id="historical-chart"',
        'id="historical-tooltip"',
        'id="historical-chart-download"',
    ):
        assert anchor in root_html, f"Missing anchor in root index.html: {anchor}"
    assert 'id="historical-chart" style=' not in root_html
