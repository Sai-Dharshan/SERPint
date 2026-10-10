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


def test_pilot_is_5_by_5_and_covers_every_tier():
    s = load_settings()
    pilot = s.plans["pilot"]
    assert pilot.size == 25
    assert {s.query(q).tier for q in pilot.queries} == set(TIERS)


def test_core_fits_free_tier_and_contains_pilot():
    s = load_settings()
    core, pilot = s.plans["core"], s.plans["pilot"]
    assert core.size == 240 <= 250
    assert set(pilot.cities) <= set(core.cities)
    assert set(pilot.queries) <= set(core.queries)
    assert {s.query(q).tier for q in core.queries} == set(TIERS)


def test_core_keeps_every_same_state_pair():
    s = load_settings()
    by_state = Counter(s.city(c).state for c in s.plans["core"].cities)
    for state in ("Tamil Nadu", "Maharashtra", "Karnataka", "Kerala"):
        assert by_state[state] == 2, state


def test_plan_validation(tmp_path):
    import shutil

    import pytest

    cfg = tmp_path / "config"
    shutil.copytree(load_settings().root / "config", cfg)
    path = cfg / "settings.yaml"
    original = path.read_text()

    path.write_text(original + "\n  bad:\n    cities: [atlantis]\n    queries: [weather]\n")
    with pytest.raises(ValueError, match="unknown ids"):
        load_settings(config_dir=cfg)

    path.write_text(original + "\n  grid:\n    cities: [delhi]\n    queries: [weather]\n")
    with pytest.raises(ValueError, match="reserved"):
        load_settings(config_dir=cfg)


def test_search_params_match_spec():
    p = load_settings().search_params
    assert (p["gl"], p["hl"], p["google_domain"]) == ("in", "en", "google.co.in")
