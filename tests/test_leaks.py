from serpint import leaks
from serpint.parse import Item, Page

CITIES = ["chennai", "mumbai", "delhi", "kolkata"]


def _page(query, city, elements):
    return Page(query, city, "neutral", "t", {k: [Item(v)] for k, v in elements.items()})


def _pages():
    """`signal` names a regional paper per city in every query; `noise` never varies.

    `once` shows up in one query only, so it can't be evaluated leave-one-query-out.
    """
    out = []
    for q in ("q1", "q2", "q3", "q4"):
        for c in CITIES:
            els = {"signal": f"{c} herald on {q}", "noise": f"generic result for {q}"}
            if q == "q1":
                els["once"] = "a block seen once"
            out.append(_page(q, c, els))
    return out


def test_keep_only_and_drop():
    page = _page("q", "delhi", {"a": "x", "b": "y"})
    assert list(leaks.keep_only(page, "a").elements) == ["a"]
    assert list(leaks.drop(page, "a").elements) == ["b"]
    assert list(page.elements) == ["a", "b"]  # the original is untouched


def test_leak_map_separates_signal_from_noise(settings):
    found, full = leaks.leak_map(_pages(), settings, n_jobs=1)
    by = {x.element: x for x in found}

    signal = by["signal"].ablations["text"]
    assert signal.alone.accuracy == 1.0
    assert signal.without.accuracy < signal.full.accuracy
    assert by["signal"].names_own_city == 1.0

    noise = by["noise"].ablations["text"]
    assert noise.alone.accuracy < 0.5
    assert noise.without.accuracy == noise.full.accuracy == 1.0

    # Masking blanks the city in "<city> herald", leaving nothing to learn from.
    assert by["signal"].ablations["masked"].alone.accuracy < 0.5

    assert by["once"].queries == 1
    assert by["once"].ablations["text"].alone is None
    assert full["text"].accuracy == 1.0


def test_top_features_point_at_the_leaking_element(settings):
    pages = [
        _page(q, c, {"signal": f"{c[:3]}paper", "noise": f"generic {q}"})
        for q in ("q1", "q2", "q3")
        for c in CITIES
    ]
    feats = leaks.top_features(pages, settings, k=2)
    assert set(feats) == set(CITIES)
    top = feats["delhi"][0]
    assert top.token == "delpaper"
    assert top.element == "signal"
    assert top.pages == 3


def test_report_marks_single_query_elements(settings):
    found, full = leaks.leak_map(_pages(), settings, n_jobs=1)
    feats = leaks.top_features(_pages(), settings)
    report = leaks.leak_report(found, full, feats, settings, 16)
    assert "| `once` | 4 | 1 | 0% | n/a | n/a |" in report
    assert report.index("`signal`") < report.index("`noise`")  # ranked by what it leaks alone
