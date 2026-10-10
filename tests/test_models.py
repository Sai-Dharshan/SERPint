import math

import numpy as np
import pytest

from serpint import models
from serpint.parse import Item, Page

CITIES = ["chennai", "mumbai", "delhi", "kolkata"]


def _page(query, city, text, domain=None, tier="neutral"):
    return Page(query, city, tier, "t", {"organic_results": [Item(text, domain)]})


@pytest.fixture
def geo(settings):
    return models.Geo.of((settings.city(c) for c in CITIES), settings.geoguessr_scale_km)


def test_mask_removes_whole_words_and_domains(settings):
    mask = models.mask_pattern(settings.cities)
    page = _page("q", "kochi", "Weather in Kochi, Keralam | flats-in-chennai", "kochimetro.org")
    doc = models.document(page, mask)
    assert doc == "Weather in PLACE, PLACE | flats-in-PLACE\ndom_PLACE_org"
    assert models.document(page).endswith("dom_kochimetro_org")


def test_document_can_keep_only_some_elements():
    page = Page("q", "pune", "neutral", "t", {"a": [Item("one")], "b": [Item("two")]})
    assert models.document(page, elements=["b"]) == "two"


def test_central_city_is_the_middle_one(settings):
    # Mumbai, Pune and Chennai: Pune sits between, nearer Mumbai.
    geo = models.Geo.of(
        [settings.city("mumbai"), settings.city("pune"), settings.city("chennai")], 400
    )
    assert geo.ids[geo.central()] == "pune"


def test_ranks_average_ties():
    assert models._ranks(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [2.5, 0.0, 2.5, 1.0]


def test_constant_model_always_picks_central(geo):
    m = models.ConstantModel().fit([], geo)
    probs = m.predict_proba([_page("q", "delhi", "x")] * 2)
    assert probs.sum(axis=1).tolist() == [1.0, 1.0]
    assert (probs.argmax(axis=1) == geo.central()).all()


def test_mention_model_counts_names_and_falls_back_to_central(geo):
    m = models.MentionModel().fit([], geo)
    named, unnamed = m.predict_proba(
        [_page("q", "delhi", "Mumbai Indians at Delhi, Delhi"), _page("q", "delhi", "eggs")]
    )
    assert geo.ids[int(named.argmax())] == "delhi"  # 2 city + 1 state hit beat Mumbai's 1
    assert math.isclose(named.sum(), 1.0)
    assert int(unnamed.argmax()) == geo.central()


def test_leave_one_query_out_never_trains_on_the_held_out_query(settings, monkeypatch):
    seen = []

    class Spy(models.ConstantModel):
        def fit(self, pages, geo):
            seen.append({p.query_id for p in pages})
            return super().fit(pages, geo)

    pages = [_page(q, c, f"{q} {c}") for q in ("q1", "q2", "q3") for c in CITIES]
    preds = models.leave_one_query_out(pages, settings, Spy)
    assert seen == [{"q2", "q3"}, {"q1", "q3"}, {"q1", "q2"}]
    assert [(p.query_id, p.city_id) for p in preds] == [(p.query_id, p.city_id) for p in pages]


def test_text_model_learns_a_signal_that_carries_across_queries(settings):
    # Each city's pages share a regional outlet whatever the query.
    pages = [
        _page(q, c, f"result about {q}", f"{c}-daily.example")
        for q in ("q1", "q2", "q3", "q4")
        for c in CITIES
    ]
    preds = models.leave_one_query_out(pages, settings, lambda: models.TextModel(seed=0))
    s = models.summarize(preds, settings)
    assert s.accuracy == 1.0
    assert s.median_km == 0.0
    assert s.mean_score == 5000.0

    masked = models.leave_one_query_out(
        pages, settings, lambda: models.TextModel(models.mask_pattern(settings.cities))
    )
    assert models.summarize(masked, settings).accuracy < 1.0  # outlet names were the signal


def test_distance_signal_detects_regional_pages_and_identical_pages(settings):
    # 6 + 6 cities: enough distinct splits for a permutation test to reach p < 0.05.
    north = {"delhi", "chandigarh", "lucknow", "jaipur", "ahmedabad", "kolkata"}
    cities = [*north, "chennai", "bengaluru", "kochi", "coimbatore", "mysuru", "hyderabad"]
    regional = [
        _page("regional", c, "north edition" if c in north else "south edition") for c in cities
    ]
    same = [_page("same", c, "identical page") for c in cities]
    by_q = {s.query_id: s for s in models.distance_signal(regional + same, settings, n_perm=199)}
    assert by_q["regional"].rho > 0.5
    assert by_q["regional"].p_value < 0.05
    assert math.isnan(by_q["same"].rho)
    assert by_q["same"].mean_text_distance == pytest.approx(0.0)


def test_report_renders(settings):
    pages = [_page(q, c, f"{c} news", tier="implicit_local") for q in ("q1", "q2") for c in CITIES]
    preds = {"mentions": models.leave_one_query_out(pages, settings, models.MentionModel)}
    signals = {"text": models.distance_signal(pages, settings, n_perm=9)}
    report = models.model_report(preds, signals, settings, len(pages))
    assert "### implicit_local" in report
    assert "| `mentions` | 8 | 100% | 100% | 0 | 5,000 |" in report
