import base64
import re
from io import BytesIO

import pytest
from PIL import Image

from serpint import app as app_module
from serpint import scrub
from serpint.parse import Item, Page

CITIES = {"chennai": "Chennai", "mumbai": "Mumbai", "delhi": "Delhi", "kolkata": "Kolkata"}

OCR_LINES = [
    scrub.Line((10, 10, 200, 30), "best biryani", 1.0),
    scrub.Line((10, 60, 300, 80), "biryani near anna nagar chennai", 1.0),
    scrub.Line((10, 400, 380, 420), "Chennai, Tamil Nadu - From your IP address", 1.0),
]


@pytest.fixture
def client(settings, monkeypatch, tmp_path):
    pages = [
        Page(
            q,
            c,
            "neutral",
            "t",
            {"o": [Item(f"{q} {name} anna" if c == "chennai" else f"{q} {name}")]},
        )
        for q in ("q1", "q2", "q3")
        for c, name in CITIES.items()
    ]
    attacker = app_module.Attacker(pages, settings)
    monkeypatch.setattr(scrub, "ocr", lambda image: OCR_LINES)
    monkeypatch.chdir(tmp_path)
    flask_app = app_module.create_app(settings, attacker)
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def _png(size=(400, 450)):
    buf = BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Check and scrub" in r.data
    assert r.headers["Cache-Control"] == "no-store"


def test_upload_is_analysed_scrubbed_and_not_stored(client, tmp_path):
    r = client.post(
        "/scrub",
        data={"image": (BytesIO(_png()), "shot.png"), "level": "names"},
        content_type="multipart/form-data",
    )
    html = r.data.decode()
    assert r.status_code == 200
    assert "Names on the page: Chennai (2)" in html
    assert "No configured city is named in what's left." in html
    assert "Google&#39;s location line: 1" in html and "names a place: 1" in html
    png = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)"', html).group(1)
    assert Image.open(BytesIO(base64.b64decode(png))).size == (400, 450)
    assert list(tmp_path.rglob("*")) == []  # nothing written anywhere under the cwd


def test_ocr_text_is_escaped(client, monkeypatch):
    evil = [scrub.Line((0, 0, 10, 10), "<script>alert(1)</script> Mumbai", 1.0)]
    monkeypatch.setattr(scrub, "ocr", lambda image: evil)
    r = client.post(
        "/scrub", data={"image": (BytesIO(_png()), "x.png")}, content_type="multipart/form-data"
    )
    assert b"<script>alert" not in r.data


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, b"Choose a screenshot"),
        ({"image": (BytesIO(b"not an image"), "x.png")}, b"readable PNG or JPEG"),
        ({"image": (BytesIO(_png()), "x.png"), "level": "all"}, b"Unknown blur level"),
    ],
)
def test_bad_uploads(client, data, message):
    r = client.post("/scrub", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert message in r.data


def test_too_large(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_PIXELS", 100)
    r = client.post(
        "/scrub", data={"image": (BytesIO(_png()), "x.png")}, content_type="multipart/form-data"
    )
    assert r.status_code == 400
    assert b"too large" in r.data
