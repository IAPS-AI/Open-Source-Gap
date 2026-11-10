"""Parsing benchmarked_models.csv and the open/closed label rules."""
import csv
import io
from datetime import date

import pytest

from accessibility import (
    ACCESSIBILITY_OVERRIDES,
    AccessibilitySourceError,
    classify_accessibility,
    is_unrecognized_label,
    report_unrecognized_labels,
)
from eci_data import decode, load_models

COLUMNS = ["Model", "id_model_version", "Organization", "Country (of organization)",
           "Version release date", "Model accessibility", "Description", "model_group", "eci",
           "eci_ci_low", "eci_ci_high"]


def sheet(*rows, columns=COLUMNS):
    """CSV text from rows given as dicts; unnamed columns are blank."""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    return out.getvalue()


def row(group, eci, day="2025-01-01", label="API access", org="Lab",
        country="United States of America", **extra):
    return {"Model": extra.pop("family", group), "model_group": group, "eci": eci,
            "Version release date": day, "Model accessibility": label,
            "Organization": org, "Country (of organization)": country, **extra}


def by_name(text):
    models, report = load_models(text)
    return {m.name: m for m in models}, report


# ------------------------------------------------------------------ labels

@pytest.mark.parametrize("label, expected", [
    ("Open weights (unrestricted)", True),
    ("Open weights (restricted use)", True),
    ("Open weights (non-commercial)", True),
    ("API access", False),
    ("Hosted access (no API)", False),
    ("Limited access", False),          # capability the closed side holds
    ("Unreleased", None),
    ("Unknown", None),
    ("Some new label", None),
    ("", None),
    (None, None),
    (float("nan"), None),
])
def test_classify_accessibility(label, expected):
    assert classify_accessibility(label) is expected


def test_only_new_labels_count_as_unrecognized():
    assert is_unrecognized_label("Some new label")
    for known in ("Open weights (unrestricted)", "API access", "Unreleased", "Unknown", "", None):
        assert not is_unrecognized_label(known)


def test_unrecognized_labels_are_flagged_on_the_actions_run(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    found = report_unrecognized_labels(["API access", "Weights on request", "", "Weights on request"], "x.csv")
    assert found == ["Weights on request"]
    assert "::warning title=Unrecognized accessibility label::" in capsys.readouterr().out


def test_every_override_is_a_label_the_classifier_understands():
    for name, label in ACCESSIBILITY_OVERRIDES.items():
        assert classify_accessibility(label) is not None, name


# ----------------------------------------------------------------- parsing

def test_versions_of_one_group_collapse_to_one_model_with_the_earliest_date():
    text = sheet(
        row("GPT-5", 150.0, "2025-08-07", id_model_version="gpt-5_high"),
        row("GPT-5", 150.0, "", id_model_version="gpt-5-chat"),                 # undated variant
        row("Gemini 2.5 Pro (Mar 2025)", 143.91, "2025-03-31", eci_ci_low=141.9, eci_ci_high=145.8),
        row("Gemini 2.5 Pro (Mar 2025)", 143.91, "2025-03-25", eci_ci_low=141.9, eci_ci_high=145.8),
    )
    models, report = by_name(text)
    assert report.scored == 2 and not report.undated
    assert models["GPT-5"].released == date(2025, 8, 7)
    assert (models["GPT-5"].ci_low, models["GPT-5"].ci_high, models["GPT-5"].sd) == (None, None, None)
    gemini = models["Gemini 2.5 Pro (Mar 2025)"]
    assert gemini.released == date(2025, 3, 25)
    assert (gemini.eci, gemini.ci_low, gemini.ci_high) == (143.91, 141.9, 145.8)
    assert gemini.organization == "Lab" and gemini.is_open is False


def test_dated_releases_of_one_family_are_separate_models():
    text = sheet(
        row("GPT-4o (May 2024)", 128.2, "2024-05-13", family="GPT-4o"),
        row("GPT-4o (Nov 2024)", 128.95, "2024-11-20", family="GPT-4o"),
    )
    models, _ = by_name(text)
    assert {n: (m.released, m.eci) for n, m in models.items()} == {
        "GPT-4o (May 2024)": (date(2024, 5, 13), 128.2),
        "GPT-4o (Nov 2024)": (date(2024, 11, 20), 128.95),
    }


def test_rows_without_a_score_are_ignored_and_undated_groups_reported():
    text = sheet(row("Scored", 120), row("Pending", ""), row("Timeless", 110, day=""))
    models, report = by_name(text)
    assert set(models) == {"Scored"}
    assert report.scored == 2 and report.undated == ["Timeless"]


def test_a_blank_group_name_falls_back_to_the_model_column():
    text = sheet({**row("", 120), "Model": "Fallback name"})
    models, _ = by_name(text)
    assert set(models) == {"Fallback name"}


def test_quoted_multi_line_fields_do_not_break_rows():
    text = sheet(row("A", 120, Description='Line one,\nline "two"\nline three'), row("B", 121))
    models, _ = by_name(text)
    assert set(models) == {"A", "B"}


def test_country_is_epochs_label_for_the_developer():
    text = sheet(row("Kimi K3", 158.5, country="China", label="Open weights (unrestricted)"),
                 row("GPT-5", 150.0),
                 row("Mistral Large", 140.0, country="France"))
    models, report = by_name(text)
    assert {name: m.country for name, m in models.items()} == {
        "Kimi K3": "China", "GPT-5": "United States of America", "Mistral Large": "France"}
    assert report.no_country == []


def test_a_few_models_without_a_country_are_kept_and_reported():
    text = sheet(row("Stateless", 120, country=""), *[row(f"Filler {i}", 100 + i) for i in range(9)])
    models, report = by_name(text)
    assert models["Stateless"].country == "" and report.no_country == ["Stateless"]


def test_decode_reads_utf8_whether_or_not_it_has_a_byte_order_mark():
    assert decode("Tülu 3".encode("utf-8")) == "Tülu 3"
    assert decode(b"\xef\xbb\xbfModel,eci") == "Model,eci"


# ------------------------------------------------------- open/closed rules

def test_open_and_closed_come_from_the_label_and_unlabelled_models_are_left_out():
    text = sheet(
        row("Open one", 120, label="Open weights (restricted use)"),
        row("Closed one", 125, label="Hosted access (no API)"),
        row("Limited one", 130, label="Limited access"),
        row("Lab-only", 135, label="Unreleased"),
        row("Mystery", 140, label=""),
        *[row(f"Filler {i}", 100 + i) for i in range(20)],     # keep label coverage healthy
    )
    models, report = by_name(text)
    assert models["Open one"].is_open is True
    assert models["Closed one"].is_open is False and models["Limited one"].is_open is False
    assert "Lab-only" not in models and "Mystery" not in models
    assert report.left_out == ["Lab-only", "Mystery"]
    assert report.scored == 25 and report.labelled == 23


def test_an_override_fills_a_blank_label_but_never_replaces_epochs(monkeypatch):
    monkeypatch.setitem(ACCESSIBILITY_OVERRIDES, "Blank", "Open weights (unrestricted)")
    monkeypatch.setitem(ACCESSIBILITY_OVERRIDES, "Labelled", "Open weights (unrestricted)")
    text = sheet(row("Blank", 120, label=""), row("Labelled", 121, label="API access"),
                 *[row(f"Filler {i}", 100 + i) for i in range(10)])
    models, report = by_name(text)
    assert models["Blank"].is_open is True
    assert models["Labelled"].is_open is False
    assert report.labelled == 11                     # the override is not Epoch's label


def test_published_weights_win_if_a_groups_versions_disagree():
    text = sheet(row("Mixed", 120, label="API access"),
                 row("Mixed", 120, label="Open weights (unrestricted)"))
    models, _ = by_name(text)
    assert models["Mixed"].is_open is True


# ------------------------------------------------------------ build guards

def test_missing_label_column_stops_the_build():
    columns = [c for c in COLUMNS if c != "Model accessibility"]
    with pytest.raises(AccessibilitySourceError, match="Model accessibility"):
        load_models(sheet(row("A", 120), columns=columns))


def test_sparse_labels_stop_the_build():
    text = sheet(row("A", 120), row("B", 121, label=""), row("C", 122, label=""))
    with pytest.raises(AccessibilitySourceError, match="33.3%"):
        load_models(text)


def test_overrides_do_not_rescue_sparse_labels(monkeypatch):
    for name in ("B", "C"):
        monkeypatch.setitem(ACCESSIBILITY_OVERRIDES, name, "API access")
    text = sheet(row("A", 120), row("B", 121, label=""), row("C", 122, label=""))
    with pytest.raises(AccessibilitySourceError):
        load_models(text)


def test_sparse_or_missing_country_labels_stop_the_build():
    columns = [c for c in COLUMNS if c != "Country (of organization)"]
    with pytest.raises(ValueError, match="Country"):
        load_models(sheet(row("A", 120), columns=columns))
    text = sheet(row("A", 120), row("B", 121, country=""), row("C", 122, country=""))
    with pytest.raises(ValueError, match="2 of 3 models have no 'Country"):
        load_models(text)


def test_missing_required_column_or_no_scores_is_an_error():
    columns = [c for c in COLUMNS if c != "model_group"]
    with pytest.raises(ValueError, match="model_group"):
        load_models(sheet(row("A", 120), columns=columns))
    with pytest.raises(ValueError, match="no rows with an ECI score"):
        load_models(sheet(row("A", "")))
