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
