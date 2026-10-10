"""Render cached results pages to PNG screenshots, offline.

Each page's raw HTML (``data/html/…``, saved during collection) is loaded in headless
Chromium with **every network request blocked**: nothing is fetched from Google or
anywhere else, and the page renders from what's in the file (Google inlines its CSS and
most images).

One thing doesn't survive going offline. Google's footer line ("Chennai, Tamil Nadu -
From your IP address - Update location") is inserted by a script that depends on
external JavaScript, so it would render empty. The text is in the HTML, though, so it's
read from there and written into the footer as plain text. Screenshots record that they
carry a reconstructed footer, and ``--no-footer`` turns the reconstruction off.
"""

from __future__ import annotations

import gzip
import html
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from serpint.client import SerpApiClient, atomic_write_bytes
from serpint.collect import plan_cells
from serpint.config import Settings

# Playwright's own browser download isn't used; the environment's Chromium is.
CHROMIUM = Path("/opt/pw-browsers/chromium")
VIEWPORT = {"width": 1366, "height": 900}  # a common laptop screen

# The footer location, as it sits (JS-escaped) in Google's HTML.
_FOOTER_PLACE = re.compile(r'class=\\"AhYzQb\\">(.*?)\\x3c/span>')
_FOOTER_SOURCE = re.compile(r'class=\\"KwU3F\\">\\x3cspan>(.*?)\\x3c/span>')

_FOOTER_JS = """
([place, source]) => {
  const foot = document.querySelector('#footcnt') || document.body.appendChild(
    document.createElement('div'));
  const bar = document.createElement('div');
  bar.setAttribute('data-serpint', 'reconstructed-footer');
  bar.style.cssText = 'background:#f2f2f2;color:#70757a;font:14px arial,sans-serif;' +
    'padding:15px 30px 15px 125px;border-top:1px solid #dadce0;line-height:22px';
  bar.innerHTML = '<div style="color:#202124">India</div><div><b>' + place +
    '</b> - ' + source + ' - <span style="color:#1a0dab">Update location</span></div>';
  foot.appendChild(bar);
}
"""


@dataclass(frozen=True)
class Shot:
    """A rendered screenshot and how it was made."""

    query_id: str
    city_id: str
    timestamp: str
    path: Path
    width: int
    height: int
    footer: str | None  # the reconstructed footer location, if one was added


def footer_location(raw_html: str) -> tuple[str, str] | None:
    """Return ``(place, source)`` of Google's footer location line, if the HTML has one."""
    place, source = _FOOTER_PLACE.search(raw_html), _FOOTER_SOURCE.search(raw_html)
    if place is None:
        return None
    unescape = lambda s: html.unescape(s.encode().decode("unicode_escape"))  # noqa: E731
    return unescape(place.group(1)), unescape(source.group(1)) if source else ""


def screenshot_dir(settings: Settings) -> Path:
    """Return where screenshots go (git-ignored; regenerate with ``serpint screenshots``)."""
    return settings.data_dir / "screenshots"


def shot_path(settings: Settings, query_id: str, city_id: str, timestamp: str) -> Path:
    """Return the PNG path for one cell."""
    return screenshot_dir(settings) / query_id / city_id / f"{timestamp}.png"


@contextmanager
def browser() -> Iterator[Any]:
    """Yield a headless Chromium (Playwright imported lazily: it's only needed here)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        kwargs = {"executable_path": str(CHROMIUM)} if CHROMIUM.exists() else {}
        b = p.chromium.launch(**kwargs)
        try:
            yield b
        finally:
            b.close()


def render(
    b: Any, raw_html: str, out: Path, full_page: bool = True, footer: bool = True
) -> tuple[int, int, str | None]:
    """Render ``raw_html`` offline to ``out``; return ``(width, height, footer place)``."""
    page = b.new_page(viewport=VIEWPORT)
    try:
        page.route("**/*", lambda route: route.abort())  # nothing leaves the machine
        page.set_content(raw_html, wait_until="load")
        place = None
        if footer and (loc := footer_location(raw_html)) is not None:
            page.evaluate(_FOOTER_JS, [html.escape(loc[0]), html.escape(loc[1])])
            place = loc[0]
        png = page.screenshot(full_page=full_page)
    finally:
        page.close()
    atomic_write_bytes(out, png)
    from PIL import Image

    with Image.open(out) as im:
        return im.width, im.height, place


def take_screenshots(
    client: SerpApiClient,
    settings: Settings,
    plan: str,
    footer: bool = True,
    refresh: bool = False,
    echo: Any = print,
) -> list[Shot]:
    """Render every cached cell of a plan that has raw HTML; skip ones already rendered."""
    shots = []
    with browser() as b:
        for c in plan_cells(settings, plan):
            hit = client.cached(c.query.id, c.city.id)
            if hit is None:
                continue
            src = client.html_path(c.query.id, c.city.id, hit.timestamp)
            if not src.exists():
                echo(f"skip {c.query.id}@{c.city.id}: no raw HTML")
                continue
            out = shot_path(settings, c.query.id, c.city.id, hit.timestamp)
            meta = out.with_suffix(".json")
            if out.exists() and meta.exists() and not refresh:
                shots.append(Shot(**{**json.loads(meta.read_text()), "path": out}))
                continue
            raw = gzip.decompress(src.read_bytes()).decode("utf-8", "replace")
            w, h, place = render(b, raw, out, footer=footer)
            shot = Shot(c.query.id, c.city.id, hit.timestamp, out, w, h, place)
            meta.write_text(json.dumps({**vars(shot), "path": None}) + "\n")
            shots.append(shot)
            echo(f"ok   {c.query.id}@{c.city.id}  {w}x{h}")
    return shots
