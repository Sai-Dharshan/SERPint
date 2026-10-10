import dataclasses
import gzip
import json

import pytest
import requests

from conftest import SECRET, FakeSession, make_response, ok_search
from serpint.client import NO_RESULTS_ERROR, BudgetExceededError, SerpApiClient, SerpApiError


@pytest.fixture
def client(settings, session):
    return SerpApiClient(settings, SECRET, session=session)


def _cell(settings, q="weather", c="chennai"):
    return settings.query(q), settings.city(c)


def test_search_caches_response_and_records_ledger(client, settings, session):
    q, c = _cell(settings)
    got = client.search(q, c, "Chennai,Tamil Nadu,India")

    assert got.path.parent == settings.data_dir / "raw" / "weather" / "chennai"
    assert got.path.suffix == ".json"
    assert json.loads(got.path.read_text())["search_metadata"]["status"] == "Success"
    events = [e["event"] for e in client.ledger.entries()]
    assert events == ["started", "finished"]
    assert client.ledger.spent() == 1

    url, params = session.calls_to("search.json")[0]
    assert params["q"] == "weather"
    assert params["location"] == "Chennai,Tamil Nadu,India"
    assert (params["gl"], params["hl"], params["google_domain"]) == ("in", "en", "google.co.in")


def test_cached_cell_is_never_refetched(client, settings, session):
    q, c = _cell(settings)
    first = client.search(q, c, "Chennai,Tamil Nadu,India")
    again = client.search(q, c, "Chennai,Tamil Nadu,India")
    assert again.path == first.path
    assert len(session.calls_to("search.json")) == 1
    assert client.ledger.spent() == 1


def test_cache_survives_a_new_client(client, settings, session):
    q, c = _cell(settings)
    client.search(q, c, "loc")
    fresh = SerpApiClient(settings, SECRET, session=session)
    fresh.search(q, c, "loc")
    assert len(session.calls_to("search.json")) == 1


def test_budget_refuses_before_calling(settings, session):
    tight = dataclasses.replace(settings, max_searches=1)
    client = SerpApiClient(tight, SECRET, session=session)
    client.search(*_cell(tight, c="chennai"), "a")
    with pytest.raises(BudgetExceededError):
        client.search(*_cell(tight, c="delhi"), "b")
    assert len(session.calls_to("search.json")) == 1
    assert client.ledger.spent() == 1


def test_allow_over_budget_is_explicit(settings, session):
    tight = dataclasses.replace(settings, max_searches=1)
    client = SerpApiClient(tight, SECRET, allow_over_budget=1, session=session)
    client.search(*_cell(tight, c="chennai"), "a")
    client.search(*_cell(tight, c="delhi"), "b")
    assert client.ledger.spent() == 2


def test_error_is_counted_not_cached_and_redacted(settings):
    s = FakeSession(lambda u, p: make_response({"error": f"Invalid API key {SECRET}"}, 401))
    client = SerpApiClient(settings, SECRET, session=s)
    with pytest.raises(SerpApiError) as e:
        client.search(*_cell(settings), "loc")
    assert SECRET not in str(e.value)
    assert not client.is_cached("weather", "chennai")
    assert client.ledger.spent() == 1  # conservative: attempts count
    finished = client.ledger.entries()[-1]
    assert finished["status"] == "error" and SECRET not in json.dumps(finished)


def test_network_exception_never_leaks_key(settings):
    def boom(url, params):
        raise requests.ConnectionError(f"failed GET {url}?api_key={params['api_key']}&q=x")

    client = SerpApiClient(settings, SECRET, session=FakeSession(boom))
    with pytest.raises(SerpApiError) as e:
        client.search(*_cell(settings), "loc")
    assert SECRET not in str(e.value)
    assert SECRET not in client.ledger.path.read_text()


def test_no_results_page_is_cached_as_an_observation(settings):
    payload = {"search_metadata": {"id": "x", "status": "Success"}, "error": NO_RESULTS_ERROR}
    session = FakeSession(lambda u, p: make_response(payload))
    client = SerpApiClient(settings, SECRET, session=session)
    client.search(*_cell(settings), "loc")
    assert client.is_cached("weather", "chennai")
    assert client.ledger.entries()[-1]["status"] == "no_results"


def test_refuses_to_write_the_key_to_disk(settings):
    payload = {"search_metadata": {"id": "x", "status": "Success"}, "echo": SECRET}
    session = FakeSession(lambda u, p: make_response(payload))
    client = SerpApiClient(settings, SECRET, session=session)
    with pytest.raises(SerpApiError):
        client.search(*_cell(settings), "loc")
    assert not client.is_cached("weather", "chennai")


def test_missing_key_fails_before_spending(settings, session):
    client = SerpApiClient(settings, None, session=session)
    with pytest.raises(SerpApiError, match="SERPAPI_API_KEY"):
        client.search(*_cell(settings), "loc")
    assert client.ledger.spent() == 0 and session.calls == []


def test_raw_html_is_free_gzipped_and_fetched_once(client, settings, session):
    hit = client.search(*_cell(settings), "loc")
    path = client.fetch_raw_html(hit, "weather", "chennai")
    assert path.name == f"{hit.timestamp}.html.gz"
    assert gzip.decompress(path.read_bytes()).startswith(b"<html>")
    client.fetch_raw_html(hit, "weather", "chennai")
    assert len([c for c in session.calls if c[0].endswith(".html")]) == 1
    assert client.ledger.spent() == 1


def test_locations_are_cached_and_free(settings):
    s = FakeSession(lambda u, p: make_response([{"canonical_name": "Pune,Maharashtra,India"}]))
    client = SerpApiClient(settings, None, session=s)
    assert client.locations("Pune")[0]["canonical_name"].startswith("Pune")
    client.locations("Pune")
    assert len(s.calls) == 1
    assert "api_key" not in s.calls[0][1]
    assert client.ledger.spent() == 0


def test_lock_blocks_a_second_run(client):
    with client.lock(), pytest.raises(RuntimeError, match="lock"), client.lock():
        pass


def test_account_reads_balance(settings):
    client = SerpApiClient(settings, SECRET, session=FakeSession(ok_search))
    assert client.account()["total_searches_left"] == 1000
