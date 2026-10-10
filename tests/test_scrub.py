import pytest
from PIL import Image, ImageDraw

from serpint import scrub
from serpint.screenshots import footer_location


def L(y, text, x=125, w=400, h=20):
    return scrub.Line((x, y, x + w, y + h), text, 0.99)


@pytest.fixture
def gaz(settings):
    return scrub.Gazetteer.load(settings)


@pytest.mark.parametrize(
    ("text", "place"),
    [
        ("Best biryani in Chennai", "Chennai"),
        ("Chennai, Tamil Nadlu - From your IP", "Chennai"),  # OCR misread elsewhere
        ("Tamil Nadlu", "Tamil Nadu"),  # fuzzy on a long name
        ("Mani'sDumBiryani-Wallajahroad,Hyderabadstyle", "Hyderabad"),  # glued
        ("weather in Bangalore", "Bangalore"),
        ("Visit Goa this winter", "Goa"),
        ("how to boil eggs", None),
        ("opportune moment", None),  # "pune" inside a word: short names need whole words
        ("Dakota and goals", None),  # same for "Kota" and "Goa"
    ],
)
def test_gazetteer(gaz, text, place):
    assert gaz.find(text) == place


def test_names_level_rules(gaz):
    lines = [
        L(10, "what is GST"),
        L(40, "Kolkata, West Bengal - From your IP address - Update location"),
        L(70, "Results for Pune, Maharashtra · Choose area"),
        L(100, "MG Road, 411001"),
        L(130, "Call 080 41234567"),
        L(160, "Price ₹76,999 for 128 GB"),  # a price is not a PIN code
    ]
    red = scrub.plan_redactions(lines, gaz, "names")
    assert [r.reason for r in red] == ["location_line", "location_line", "pin_code", "landline"]
    kept = scrub.kept_lines(lines, red)
    assert [ln.text for ln in kept] == ["what is GST", "Price ₹76,999 for 128 GB"]
    assert scrub.plan_redactions(lines, gaz, "none") == []


def test_unknown_level_is_rejected(gaz):
    with pytest.raises(ValueError, match="sections"):
        scrub.plan_redactions([], gaz, "everything")


def test_related_searches_section_runs_to_pagination(gaz):
    lines = [
        L(100, "Some result about tax"),
        L(300, "People also search for"),
        L(350, "gst in banking", w=200),
        L(350, "gst near anna nagar", x=460, w=250),  # second grid column
        L(410, "gst class 12", w=200),
        L(530, "1 2 3 4 5 6 7 8 9 10", x=330, w=190),
        L(530, "Next", x=578, w=30),
        L(700, "India"),
    ]
    sec = [r for r in scrub.plan_redactions(lines, gaz, "sections") if r.reason.startswith("sec")]
    assert len(sec) == 1
    assert sec[0].reason == "section:related_searches"
    x0, y0, x1, y1 = sec[0].box
    assert (y0, y1) == (300, 550)  # heading through the pagination row
    assert x1 >= 710  # covers the second column too
    assert "gst near anna nagar" not in [ln.text for ln in scrub.kept_lines(lines, sec)]
    assert "Some result about tax" in [ln.text for ln in scrub.kept_lines(lines, sec)]


def test_local_section_spans_the_map_gap(gaz):
    lines = [
        L(500, "Places"),
        L(700, "Koyambedu", x=300, w=80),  # a map label after a gap with no text
        L(900, "Mount Road Bilal"),
        L(1160, "More places", x=270, w=110),
        L(1300, "People also ask"),
    ]
    sec = [r for r in scrub.plan_redactions(lines, gaz, "sections") if r.reason.startswith("sec")]
    assert sec[0].reason == "section:local_results"
    assert (sec[0].box[1], sec[0].box[3]) == (500, 1180)


def test_apply_blurs_text_and_fills_sections():
    im = Image.new("RGB", (300, 200), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((10, 10, 100, 30), fill="black")  # stand-in for text
    d.rectangle((10, 100, 290, 190), fill=(0, 120, 255))  # stand-in for a map
    red = [
        scrub.Redaction((10, 10, 100, 30), "place_name", "x"),
        scrub.Redaction((10, 100, 290, 190), "section:local_results", "Places"),
    ]
    out = scrub.apply(im, red)
    assert out.size == im.size and im.getpixel((50, 20)) == (0, 0, 0)  # input untouched
    assert out.getpixel((50, 20)) != (0, 0, 0)  # blurred towards the white around it
    assert out.getpixel((150, 150)) == scrub.SECTION_FILL  # map gone entirely
    assert out.getpixel((200, 50)) == (255, 255, 255)  # elsewhere unchanged


def test_blurred_fraction_counts_overlaps_once():
    red = [
        scrub.Redaction((0, 0, 50, 100), "a", ""),
        scrub.Redaction((0, 0, 50, 100), "b", ""),
    ]
    frac = scrub.blurred_fraction((100, 100), red)
    assert frac == pytest.approx(54 * 100 / 10_000)  # padded by PAD on the open side


def test_footer_location_is_read_from_js_escaped_html():
    raw = (
        r"\x3cspan class=\"AhYzQb\">Chennai, Tamil Nadu\x3c/span>\x3c/div>"
        r"\x3cspan class=\"KwU3F\">\x3cspan>From your IP address\x3c/span>"
    )
    assert footer_location(raw) == ("Chennai, Tamil Nadu", "From your IP address")
    assert footer_location("<html></html>") is None
