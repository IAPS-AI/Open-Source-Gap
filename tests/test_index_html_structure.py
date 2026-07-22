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


TABLE_SORT_ANCHORS = (
    'data-sort-key="model"',
    'data-sort-key="date"',
    'data-sort-key="score"',
    'data-sort-key="type"',
    'data-sort-key="org"',
)


def test_table_sortable_headers_present(client):
    """The raw-data table headers carry data-sort-key attributes the JS
    sorting relies on, including the two new match columns."""
    html = client.get("/").get_data(as_text=True)
    for anchor in TABLE_SORT_ANCHORS:
        assert anchor in html, f"Missing anchor: {anchor}"


def test_table_sortable_headers_in_root_index():
    root_html = (Path(__file__).parent.parent / "index.html").read_text(
        encoding="utf-8"
    )
    for anchor in TABLE_SORT_ANCHORS:
        assert anchor in root_html, f"Missing anchor in root index.html: {anchor}"


def test_table_sorting_wired_in_js():
    js = (Path(__file__).parent.parent / "static" / "script.js").read_text(
        encoding="utf-8"
    )
    assert "function setupTableSorting" in js
    assert "setupTableSorting()" in js
    assert "tableSort" in js


CAVEATS_ANCHOR = 'id="caveats-panel"'


def test_caveats_panel_present(client):
    """The standing 'what this measures' caveats panel must exist in the
    served template."""
    html = client.get("/").get_data(as_text=True)
    assert CAVEATS_ANCHOR in html


def test_caveats_panel_in_root_index():
    root_html = (Path(__file__).parent.parent / "index.html").read_text(
        encoding="utf-8"
    )
    assert CAVEATS_ANCHOR in root_html


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
