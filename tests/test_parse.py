import pytest

from conftest import SECRET
from serpint import parse
from serpint.client import SerpApiClient

CHENNAI_URL = "https://serpapi.com/search.json?location=Chennai%2C+Tamil+Nadu%2C+India&q=gst"


def _page(data, city="chennai", query="what_is_gst", tier="neutral"):
    return parse.parse_response(data, query, city, tier, "20261010T000000Z")


def test_request_echo_and_urls_never_reach_visible_text():
    page = _page(
        {
            "search_parameters": {"location_requested": "Chennai,Tamil Nadu,India"},
            "search_metadata": {"google_url": "https://www.google.co.in/search?uule=Chennai"},
            "serpapi_pagination": {"next": CHENNAI_URL},
            "pagination": {"next": "https://www.google.co.in/search?q=gst&near=Chennai"},
            "related_searches": [{"query": "gst rate", "link": CHENNAI_URL}],
            "filters": [{"name": "Images", "serpapi_link": CHENNAI_URL}],
        }
    )
    assert "chennai" not in page.text().lower()
    assert page.text() == "gst rate\nImages"


def test_displayed_link_kept_link_reduced_to_domain():
    page = _page(
        {
            "organic_results": [
                {
                    "position": 1,
                    "title": "Flats for rent",
                    "link": "https://www.magicbricks.com/flats-in-chennai",
                    "displayed_link": "https://www.magicbricks.com › flats-in-chennai",
                    "about_this_result": {"regions": ["IN"]},
                    "snippet_highlighted_words": ["rent"],
                },
                {"title": "Map", "link": "https://www.google.co.in/maps/place/x"},
            ]
        }
    )
    first, second = page.elements["organic_results"]
    assert first.text == "Flats for rent | https://www.magicbricks.com › flats-in-chennai"
    assert first.domain == "magicbricks.com"
    assert second.domain is None


def test_enum_types_dropped_prose_types_kept():
    page = _page(
        {
            "answer_box": {"type": "weather_result", "location": "Kochi, Keralam"},
            "local_results": {
                "places": [
                    {"title": "Habitat", "type": "Furnished apartment building", "rating": 4.7},
                    {"title": "Cafe", "place_id": "123", "gps_coordinates": {"latitude": 8.5}},
                ]
            },
            "local_map": {"gps_coordinates": {"latitude": 8.5}, "image": "https://x/map.png"},
        }
    )
    assert page.elements["answer_box"][0].text == "Kochi, Keralam"
    assert [i.text for i in page.elements["local_results"]] == [
        "Habitat | Furnished apartment building | 4.7",
        "Cafe",
    ]
    assert page.deferred == ("local_map",)
    assert "local_map" not in page.elements


def test_results_for_line_is_a_visible_element():
    page = _page({"search_information": {"results_for": "Pune, Maharashtra", "total_results": 9}})
    assert page.text() == "Results for Pune, Maharashtra"


def test_filter_dropdown_options_are_hidden():
    page = _page({"filters": [{"name": "Cuisine", "options": [{"name": "Chettinad"}]}]})
    assert page.text() == "Cuisine"


def test_unknown_elements_still_go_through_the_filters():
    page = _page({"brand_new_block": [{"label": "Visible", "url": CHENNAI_URL, "flag": True}]})
    assert page.text() == "Visible"


def test_json_round_trip(tmp_path):
    page = _page({"organic_results": [{"title": "A", "link": "https://a.org/"}], "local_map": {}})
    path = tmp_path / "pages.jsonl"
    assert parse.write_pages([page, page], path) == 2
    assert parse.read_pages(path) == [page, page]


@pytest.mark.parametrize(
    ("city", "text", "kind"),
    [
        ("kochi", "Weather in Kochi, Keralam", (True, True, False)),
        ("bengaluru", "Best biryani in Bangalore", (True, False, False)),
        ("pune", "Punjab cricket", (False, False, False)),
        ("chennai", "Mumbai Indians beat Chennai Super Kings", (True, False, True)),
        ("delhi", "Delhi Capitals", (True, True, False)),
    ],
)
def test_mentions(settings, city, text, kind):
    patterns = parse.mention_patterns(settings.cities)
    assert parse._mention_kind(text, city, patterns) == kind


def test_report_counts_own_and_other_city(settings):
    pages = [
        _page({"related_searches": [{"query": "gst near chennai"}]}, city="chennai"),
        _page({"related_searches": [{"query": "gst in mumbai"}]}, city="pune"),
    ]
    report = parse.parse_report(pages, settings)
    assert "| neutral | 2 | 50% | 0% | 50% |" in report
    assert "| `related_searches` | 2/2 |" in report


def test_parse_plan_lists_uncollected_cells(settings, session):
    client = SerpApiClient(settings, SECRET, session=session)
    q, c = settings.query("weather"), settings.city("chennai")
    client.search(q, c, "Chennai,Tamil Nadu,India")
    result = parse.parse_plan(client, settings, "pilot")
    assert [(p.query_id, p.city_id) for p in result.pages] == [("weather", "chennai")]
    assert len(result.missing) == 24
