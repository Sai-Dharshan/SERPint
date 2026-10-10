"""Parse cached SerpAPI responses into what a person would actually see on the page.

A model may only learn from what a screenshot shows. SerpAPI responses also carry the
request itself (``search_parameters``) and many URLs that embed ``location=<city>``
(pagination, related searches, filters), so the parser keeps visible text only:

- top-level keys that never render (request echo, metadata, pagination) are dropped;
- at any depth, fields that hold URLs, ids, images or layout hints are dropped;
- any remaining string that is a URL is dropped, except ``displayed_link``, which is the
  breadcrumb Google prints under a result;
- ``type`` fields are kept only when they read as prose ("Furnished apartment building"),
  not SerpAPI enums ("weather_result").

Each page becomes a ``Page``: for every element (``organic_results``, ``top_stories``…)
a list of ``Item``s with the element's visible text and, for results, the domain.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from serpint.client import SerpApiClient
from serpint.collect import plan_cells
from serpint.config import TIERS, City, Settings

# Top-level keys that are not on the page at all.
HIDDEN_ELEMENTS = frozenset(
    {
        "search_metadata",
        "search_parameters",
        "search_information",  # only its `results_for` line renders; handled below
        "pagination",
        "serpapi_pagination",
        "top_stories_link",
        "top_stories_serpapi_link",
    }
)

# Field names (at any depth) whose values are never visible text.
_HIDDEN_FIELD = re.compile(
    r"link|thumbnail|logo|image|favicon|clip|token|kgmid|gps|uds|parameters|serpapi|"
    r"position|index|extracted_|original|place_id|provider_id|^id$|_id$|"
    r"about_this_result|about_page|highlighted_words"
)

# Per-element fields that exist in the response but sit behind a click (dropdowns,
# menus), so a screenshot of the page doesn't show them.
_HIDDEN_BY_ELEMENT = {"filters": frozenset({"options"})}

# The breadcrumb Google prints under a result: visible, despite looking like a link.
DISPLAYED_LINK = "displayed_link"

# A SerpAPI enum such as "weather_result" or "ai_overview", as opposed to a visible label.
_ENUM = re.compile(r"^[a-z0-9_]+$")
_URL = re.compile(r"^https?://", re.IGNORECASE)

# The line Google prints for the location it used ("Results for Chennai, Tamil Nadu").
RESULTS_FOR = "results_for"


@dataclass(frozen=True)
class Item:
    """One visible unit of an element: a result, a story, a place or a question."""

    text: str
    domain: str | None = None


@dataclass(frozen=True)
class Page:
    """The visible content of one results page."""

    query_id: str
    city_id: str
    tier: str
    timestamp: str
    elements: dict[str, list[Item]]
    # Elements on the page whose text SerpAPI didn't inline (e.g. an AI overview that
    # needs a follow-up request). Present, but with no text to learn from.
    deferred: tuple[str, ...] = ()

    def text(self, elements: Iterable[str] | None = None) -> str:
        """Return the visible text of the page, or of only the given elements."""
        keep = set(self.elements) if elements is None else set(elements)
        return "\n".join(
            item.text for name, items in self.elements.items() if name in keep for item in items
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "query_id": self.query_id,
            "city_id": self.city_id,
            "tier": self.tier,
            "timestamp": self.timestamp,
            "elements": {
                name: [{"text": i.text, "domain": i.domain} for i in items]
                for name, items in self.elements.items()
            },
            "deferred": list(self.deferred),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Page:
        """Rebuild a page from ``to_json`` output."""
        return cls(
            query_id=d["query_id"],
            city_id=d["city_id"],
            tier=d["tier"],
            timestamp=d["timestamp"],
            elements={
                name: [Item(i["text"], i.get("domain")) for i in items]
                for name, items in d["elements"].items()
            },
            deferred=tuple(d.get("deferred", ())),
        )


# ------------------------------------------------------------------ parsing


def _scalar_text(key: str | None, value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if _URL.match(text) and key != DISPLAYED_LINK:
        return None
    if key == "type" and _ENUM.match(text):
        return None
    return text


def visible_strings(value: Any, element: str = "", key: str | None = None) -> Iterator[str]:
    """Yield the visible text inside ``value``, depth first, in response order.

    Args:
        value: Any part of a SerpAPI response.
        element: The top-level element it belongs to (for per-element hidden fields).
        key: The field name ``value`` was found under, if any.
    """
    if isinstance(value, dict):
        hidden = _HIDDEN_BY_ELEMENT.get(element, frozenset())
        for k, v in value.items():
            if k in hidden or (k != DISPLAYED_LINK and _HIDDEN_FIELD.search(k)):
                continue
            yield from visible_strings(v, element, k)
    elif isinstance(value, list):
        for v in value:
            yield from visible_strings(v, element, key)
    else:
        text = _scalar_text(key, value)
        if text is not None:
            yield text


def _domain(item: Any) -> str | None:
    if not isinstance(item, dict) or not isinstance(item.get("link"), str):
        return None
    host = (urlsplit(item["link"]).hostname or "").lower()
    if not host or host.startswith("www.google.") or host.endswith(".gstatic.com"):
        return None
    return host.removeprefix("www.")


def _items(name: str, value: Any) -> list[Any]:
    """Split an element into its visible units."""
    if name == "local_results" and isinstance(value, dict):
        return list(value.get("places") or [])
    if isinstance(value, list):
        return value
    return [value]


def parse_response(
    data: dict[str, Any], query_id: str, city_id: str, tier: str, timestamp: str
) -> Page:
    """Turn one cached SerpAPI response into the ``Page`` a person would see."""
    elements: dict[str, list[Item]] = {}
    deferred: list[str] = []

    results_for = (data.get("search_information") or {}).get(RESULTS_FOR)
    if isinstance(results_for, str) and results_for.strip():
        elements[RESULTS_FOR] = [Item(f"Results for {results_for.strip()}")]

    for name, value in data.items():
        if name in HIDDEN_ELEMENTS:
            continue
        items = []
        for raw in _items(name, value):
            text = " | ".join(visible_strings(raw, name))
            if text:
                items.append(Item(text, _domain(raw)))
        if items:
            elements[name] = items
        else:
            # On the page, but nothing textual to read from the response (a map tile,
            # an AI overview fetched lazily).
            deferred.append(name)
    return Page(query_id, city_id, tier, timestamp, elements, tuple(deferred))


def write_pages(pages: Iterable[Page], path: Path) -> int:
    """Write pages as JSON lines; return how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_pages(path: Path) -> list[Page]:
    """Read pages written by ``write_pages``."""
    with path.open(encoding="utf-8") as f:
        return [Page.from_json(json.loads(line)) for line in f if line.strip()]


# ----------------------------------------------------------------- mentions


@dataclass(frozen=True)
class MentionPatterns:
    """Regexes that find a city's own names, and its state's, in visible text."""

    city: re.Pattern[str]
    state: re.Pattern[str]


def _names_pattern(names: Iterable[str]) -> re.Pattern[str]:
    # Anchored at a word start but open at the end, so "Keralam" (Google's spelling),
    # "Mumbaikar" and "Bengaluru's" all count.
    alts = sorted({re.escape(n) for n in names}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alts) + r")", re.IGNORECASE)


def mention_patterns(cities: Iterable[City]) -> dict[str, MentionPatterns]:
    """Build mention regexes for each city."""
    return {
        c.id: MentionPatterns(_names_pattern({c.name, *c.aliases}), _names_pattern({c.state}))
        for c in cities
    }


@dataclass
class _Tally:
    pages: int = 0
    own_city: int = 0
    own_state: int = 0
    other_city: int = 0


def _mention_kind(
    text: str, city_id: str, patterns: dict[str, MentionPatterns]
) -> tuple[bool, bool, bool]:
    own = patterns[city_id]
    own_city = bool(own.city.search(text))
    own_state = bool(own.state.search(text))
    other_city = any(p.city.search(text) for cid, p in patterns.items() if cid != city_id)
    return own_city, own_state, other_city


# ------------------------------------------------------------------- report


def _pct(n: int, d: int) -> str:
    return f"{n / d:.0%}" if d else "-"


def parse_report(pages: list[Page], settings: Settings) -> str:
    """Summarize parsed pages as Markdown.

    Covers what each element contributes (pages, items, characters, deferred) and the
    simplest leak there is: how often an element's visible text names the page's own
    city or state outright, next to how often it names some other configured city.
    """
    n = len(pages)
    patterns = mention_patterns(settings.cities)
    tiers = [t for t in TIERS if any(p.tier == t for p in pages)]

    present: Counter[str] = Counter()
    deferred: Counter[str] = Counter()
    n_items: dict[str, list[int]] = defaultdict(list)
    n_chars: dict[str, list[int]] = defaultdict(list)
    by_elem: dict[tuple[str, str], _Tally] = defaultdict(_Tally)
    by_page: dict[str, _Tally] = defaultdict(_Tally)
    by_query: dict[str, _Tally] = defaultdict(_Tally)

    for p in pages:
        deferred.update(p.deferred)
        for name, items in p.elements.items():
            present[name] += 1
            n_items[name].append(len(items))
            text = "\n".join(i.text for i in items)
            n_chars[name].append(len(text))
            t = by_elem[(name, p.tier)]
            t.pages += 1
            c, s, o = _mention_kind(text, p.city_id, patterns)
            t.own_city += c
            t.own_state += s
            t.other_city += o
        c, s, o = _mention_kind(p.text(), p.city_id, patterns)
        for t in (by_page[p.tier], by_query[p.query_id]):
            t.pages += 1
            t.own_city += c
            t.own_state += s
            t.other_city += o

    order = sorted(present, key=lambda k: (-present[k], k))
    out = [
        f"# Parsed pages ({n})",
        "",
        "Generated by `serpint parse` from cached SerpAPI responses. Only visible text is",
        "kept: request parameters, metadata and every URL (several embed `location=`) are",
        "dropped, except the `displayed_link` breadcrumb Google prints under a result.",
        "",
        "## Elements",
        "",
        "| element | pages | items (median / max) | characters (median / max) |",
        "|---|---|---|---|",
    ]
    for k in order:
        it, ch = n_items[k], n_chars[k]
        out.append(
            f"| `{k}` | {present[k]}/{n} | {statistics.median(it):g} / {max(it)} "
            f"| {statistics.median(ch):g} / {max(ch)} |"
        )
    if deferred:
        out += [
            "",
            "On the page but with no text in the response (left to the screenshot phase): "
            + ", ".join(f"`{k}` ({v})" for k, v in sorted(deferred.items())),
        ]

    out += [
        "",
        "## Explicit mentions",
        "",
        "Share of pages whose visible text names the page's own city (any alias), its own",
        "state, or a different configured city. Own-city mentions are the trivial leak a",
        "scrubber must catch; other-city mentions are noise a model has to see through.",
        "",
        "### By tier",
        "",
        "| tier | pages | own city | own state | other city |",
        "|---|---|---|---|---|",
    ]
    for tier in tiers:
        t = by_page[tier]
        out.append(
            f"| {tier} | {t.pages} | {_pct(t.own_city, t.pages)} | "
            f"{_pct(t.own_state, t.pages)} | {_pct(t.other_city, t.pages)} |"
        )

    out += ["", "### By query", "", "| query | tier | own city | own state | other city |"]
    out.append("|---|---|---|---|---|")
    for q in settings.queries:
        t = by_query.get(q.id)
        if t is None:
            continue
        out.append(
            f"| {q.id} | {q.tier} | {_pct(t.own_city, t.pages)} | "
            f"{_pct(t.own_state, t.pages)} | {_pct(t.other_city, t.pages)} |"
        )

    out += [
        "",
        "### By element and tier",
        "",
        "Own-city mention rate among pages that have the element (pages in brackets).",
        "",
        "| element | " + " | ".join(tiers) + " |",
        "|---|" + "---|" * len(tiers),
    ]
    for k in order:
        cells = []
        for tier in tiers:
            t = by_elem.get((k, tier))
            cells.append("" if t is None else f"{_pct(t.own_city, t.pages)} ({t.pages})")
        out.append(f"| `{k}` | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


@dataclass
class ParseResult:
    """Pages parsed for a plan, and the cells that had no cached response."""

    pages: list[Page] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def parse_plan(client: SerpApiClient, settings: Settings, plan: str) -> ParseResult:
    """Parse the latest cached response of every cell in a plan."""
    result = ParseResult()
    for c in plan_cells(settings, plan):
        hit = client.cached(c.query.id, c.city.id)
        if hit is None:
            result.missing.append(f"{c.query.id}@{c.city.id}")
            continue
        result.pages.append(
            parse_response(hit.data, c.query.id, c.city.id, c.query.tier, hit.timestamp)
        )
    return result
