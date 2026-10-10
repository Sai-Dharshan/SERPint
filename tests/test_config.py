from collections import Counter

from serpint.config import TIERS, load_settings


def test_grid_is_20_by_20():
    s = load_settings()
    assert len(s.cities) == 20
    assert len(s.queries) == 20


def test_every_tier_used_and_valid():
    tiers = Counter(q.tier for q in load_settings().queries)
    assert set(tiers) == set(TIERS)


def test_same_state_pairs_present():
    s = load_settings()
    for a, b in [("chennai", "coimbatore"), ("mumbai", "pune"), ("bengaluru", "mysuru")]:
        assert s.city(a).state == s.city(b).state


def test_coordinates_inside_india():
    for c in load_settings().cities:
        assert 6 < c.lat < 37 and 68 < c.lon < 98, c.id


def test_pilot_is_5_by_5_subset_of_grid():
    s = load_settings()
    assert len(s.pilot_cities) == 5 and len(s.pilot_queries) == 5
    assert {s.query(q).tier for q in s.pilot_queries} == set(TIERS)


def test_search_params_match_spec():
    p = load_settings().search_params
    assert (p["gl"], p["hl"], p["google_domain"]) == ("in", "en", "google.co.in")
