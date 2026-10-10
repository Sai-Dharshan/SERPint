import dataclasses

import pytest

from conftest import SECRET, FakeSession, make_response, ok_search
from serpint import collect
from serpint.client import SerpApiClient

LOC = {
    "Chennai": [
        {
            "canonical_name": "Chennai,Tamil Nadu,India",
            "country_code": "IN",
            "target_type": "City",
            "reach": 10,
            "gps": [80.27, 13.08],
            "google_id": 1,
        },
    ],
}


@pytest.fixture
def client(settings, session):
    return SerpApiClient(settings, SECRET, session=session)


def _locations(settings):
    return {
        c.id: collect.ResolvedLocation(
            c.id, f"{c.name},{c.state},India", c.name, None, None, None, "locations_api"
        )
        for c in settings.cities
    }


def test_plans_are_query_major(settings):
    pilot = collect.plan_cells(settings, "pilot")
    assert len(pilot) == 25
    assert [c.query.id for c in pilot[:5]] == [settings.plans["pilot"].queries[0]] * 5
    assert len(collect.plan_cells(settings, "grid")) == 400


def test_plans_nest_pilot_core_grid(settings):
    def ids(plan):
        return {(c.query.id, c.city.id) for c in collect.plan_cells(settings, plan)}

    assert len(ids("core")) == 240
    assert ids("pilot") <= ids("core") <= ids("grid")


def test_unknown_plan_is_rejected(settings):
    with pytest.raises(ValueError, match="pilot"):
        collect.plan_cells(settings, "everything")


def test_pilot_cells_are_free_when_collecting_core(client, settings, session):
    pilot = collect.plan_cells(settings, "pilot")
    for c in pilot:
        client.search(c.query, c.city, "x")
    core = collect.plan_cells(settings, "core")
    _, budget, _ = collect.describe_plan(client, core, {}, account_left=None)
    assert budget.planned_new == 240 - 25


def test_choose_location_prefers_in_state_city_then_reach(settings):
    city = settings.city("pune")
    candidates = [
        {
            "canonical_name": "Pune,Maharashtra,India",
            "country_code": "IN",
            "target_type": "City",
            "reach": 5,
        },
        {
            "canonical_name": "Pune District,Maharashtra,India",
            "country_code": "IN",
            "target_type": "District",
            "reach": 99,
        },
        {
            "canonical_name": "Pune,Other State,India",
            "country_code": "IN",
            "target_type": "City",
            "reach": 50,
        },
        {
            "canonical_name": "Pune,Somewhere,US",
            "country_code": "US",
            "target_type": "City",
            "reach": 999,
        },
    ]
    assert collect.choose_location(city, candidates)["canonical_name"] == "Pune,Maharashtra,India"
    assert collect.choose_location(city, candidates[1:2]) is None


def test_resolve_uses_aliases_and_flags_distance(settings):
    def handler(url, params):
        if params["q"] == "Chennai":
            return make_response(LOC["Chennai"])
        # Everything else resolves to a far-away point to exercise the sanity flag.
        name = params["q"]
        return make_response(
            [
                {
                    "canonical_name": f"{name},X,India",
                    "country_code": "IN",
                    "target_type": "City",
                    "reach": 1,
                    "gps": [0.0, 0.0],
                }
            ]
        )

    client = SerpApiClient(settings, None, session=FakeSession(handler))
    resolved = collect.resolve_locations(client, settings)
    assert resolved["chennai"].location == "Chennai,Tamil Nadu,India"
    assert resolved["chennai"].distance_km < 5 and not resolved["chennai"].suspicious
    assert resolved["pune"].suspicious
    assert collect.load_locations(settings)["chennai"] == resolved["chennai"]


def test_manual_location_wins(settings):
    cities = tuple(
        dataclasses.replace(c, location="Pinned,India") if c.id == "delhi" else c
        for c in settings.cities
    )
    s = dataclasses.replace(settings, cities=cities)
    handler = lambda u, p: make_response(  # noqa: E731
        [{"canonical_name": f"{p['q']},S,India", "country_code": "IN", "target_type": "City"}]
    )
    resolved = collect.resolve_locations(SerpApiClient(s, None, session=FakeSession(handler)), s)
    assert resolved["delhi"].location == "Pinned,India" and resolved["delhi"].source == "manual"


def test_describe_plan_counts_cached_and_budget(client, settings):
    cells = collect.plan_cells(settings, "pilot")
    locs = _locations(settings)
    client.search(cells[0].query, cells[0].city, "x")
    text, budget, to_fetch = collect.describe_plan(client, cells, locs, account_left=None)
    assert budget.planned_new == 24 and budget.spent == 1 and budget.fits
    assert "cached (free)" in text and "NEW (1 search)" in text

    _, budget, to_fetch = collect.describe_plan(client, cells, locs, account_left=3, limit=None)
    assert not budget.fits and budget.allowed == 3

    _, budget, to_fetch = collect.describe_plan(client, cells, locs, account_left=3, limit=3)
    assert budget.fits and len(to_fetch) == 3


def test_dry_run_style_planning_makes_no_requests(client, settings, session):
    collect.describe_plan(client, collect.plan_cells(settings, "grid"), {}, None)
    assert session.calls == []


def test_run_is_resumable_and_continues_past_failures(settings):
    def flaky(url, params):
        if "search.json" in url and params["location"].startswith("Delhi"):
            return make_response({"error": "Internal error"}, 500)
        return ok_search(url, params)

    session = FakeSession(flaky)
    client = SerpApiClient(settings, SECRET, session=session)
    cells = collect.plan_cells(settings, "pilot")
    locs = _locations(settings)
    _, _, to_fetch = collect.describe_plan(client, cells, locs, None)
    report = collect.run(client, cells, locs, to_fetch, echo=lambda s: None)
    assert len(report.fetched) == 20 and len(report.failed) == 5
    assert all(tag.endswith("@delhi") for tag in report.failed)

    # Second run: only the five failed cells are attempted again.
    session.handler = ok_search
    before = len(session.calls_to("search.json"))
    _, budget, to_fetch = collect.describe_plan(client, cells, locs, None)
    assert budget.planned_new == 5
    report = collect.run(client, cells, locs, to_fetch, echo=lambda s: None)
    assert len(session.calls_to("search.json")) - before == 5
    assert len(report.cached) == 20 and len(report.fetched) == 5
    assert client.ledger.spent() == 30  # 25 + 5 retries; failed attempts are counted


def test_run_stops_at_budget(settings, session):
    tight = dataclasses.replace(settings, max_searches=2)
    client = SerpApiClient(tight, SECRET, session=session)
    cells = collect.plan_cells(tight, "pilot")
    report = collect.run(client, cells, _locations(tight), cells, echo=lambda s: None)
    assert len(report.fetched) == 2 and len(report.failed) == 1
    assert client.ledger.spent() == 2


def test_field_inventory_reports_keys_and_presence(settings):
    rows = [
        (
            "weather",
            "chennai",
            {
                "organic_results": [{"link": "a", "title": "t"}],
                "answer_box": {"type": "weather_result"},
            },
        ),
        ("weather", "delhi", {"organic_results": [{"link": "b"}]}),
    ]
    md = collect.field_inventory(rows, settings)
    assert "| `organic_results` | 2/2 | list | 1 / 1 |" in md
    assert "| `answer_box` | 1/2 | dict |" in md
    assert "`link` 100%" in md and "`title` 50%" in md
