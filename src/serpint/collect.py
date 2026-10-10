"""Plan and run data collection: resolve locations, walk the grid, inventory the fields."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from serpint.client import (
    BudgetExceededError,
    SerpApiClient,
    SerpApiError,
    atomic_write_bytes,
)
from serpint.config import City, Query, Settings
from serpint.geo import haversine_km

# A resolved location further than this from the configured coordinates is flagged.
LOCATION_SANITY_KM = 50.0

Printer = Callable[[str], None]


# ---------------------------------------------------------------- locations


@dataclass(frozen=True)
class ResolvedLocation:
    """The SerpAPI canonical location chosen for a city."""

    city_id: str
    location: str
    alias: str | None
    google_id: int | None
    reach: int | None
    distance_km: float | None
    source: str  # "locations_api" or "manual"

    @property
    def suspicious(self) -> bool:
        """Return whether the resolved point is implausibly far from the configured city."""
        return self.distance_km is not None and self.distance_km > LOCATION_SANITY_KM


def _candidate_distance_km(candidate: dict[str, Any], city: City) -> float | None:
    """Distance from a Locations API candidate's ``gps`` to the configured city.

    SerpAPI documents ``gps`` as ``[longitude, latitude]``.
    """
    gps = candidate.get("gps")
    if not (isinstance(gps, list) and len(gps) == 2):
        return None
    lon, lat = float(gps[0]), float(gps[1])
    return haversine_km(city.lat, city.lon, lat, lon)


def choose_location(city: City, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the best Locations API candidate for ``city``.

    Keeps Indian ``City`` targets, prefers those whose canonical name includes the city's
    state, then the one with the largest reach.
    """
    cities = [
        c for c in candidates if c.get("country_code") == "IN" and c.get("target_type") == "City"
    ]
    if not cities:
        return None
    in_state = [c for c in cities if city.state in str(c.get("canonical_name", "")).split(",")]
    pool = in_state or cities
    return max(pool, key=lambda c: c.get("reach") or 0)


def load_locations(settings: Settings) -> dict[str, ResolvedLocation]:
    """Load previously resolved locations (empty if none yet)."""
    path = settings.data_dir / "locations.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {k: ResolvedLocation(**v) for k, v in json.load(f).items()}


def resolve_locations(
    client: SerpApiClient, settings: Settings, refresh: bool = False
) -> dict[str, ResolvedLocation]:
    """Resolve every city to a SerpAPI canonical location and save ``data/locations.json``.

    Uses the free Locations API (responses cached under ``data/locations_raw``). A city's
    manual ``location`` in ``cities.yaml`` always wins.

    Raises:
        SerpApiError: If a city cannot be resolved; pin it with ``location:`` in cities.yaml.
    """
    resolved = {} if refresh else load_locations(settings)
    for city in settings.cities:
        if city.location:
            resolved[city.id] = ResolvedLocation(
                city.id, city.location, None, None, None, None, "manual"
            )
            continue
        if city.id in resolved and resolved[city.id].source == "locations_api":
            continue
        for alias in city.aliases:
            best = choose_location(city, client.locations(alias))
            if best is not None:
                dist = _candidate_distance_km(best, city)
                resolved[city.id] = ResolvedLocation(
                    city_id=city.id,
                    location=str(best["canonical_name"]),
                    alias=alias,
                    google_id=best.get("google_id"),
                    reach=best.get("reach"),
                    distance_km=None if dist is None else round(dist, 1),
                    source="locations_api",
                )
                break
        else:
            raise SerpApiError(
                f"Could not resolve {city.name}; set `location:` for it in config/cities.yaml"
            )
    payload = {k: vars(v) for k, v in sorted(resolved.items())}
    atomic_write_bytes(
        settings.data_dir / "locations.json",
        (json.dumps(payload, indent=1, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    return resolved


def format_locations(settings: Settings, resolved: dict[str, ResolvedLocation]) -> str:
    """Render the resolved locations as a plain-text table."""
    lines = [f"{'city':<20} {'canonical location':<45} {'reach':>12} {'km off':>7}  note"]
    for city in settings.cities:
        r = resolved.get(city.id)
        if r is None:
            lines.append(f"{city.id:<20} {'<unresolved>':<45}")
            continue
        km = "" if r.distance_km is None else f"{r.distance_km:.1f}"
        reach = "" if r.reach is None else f"{r.reach:,}"
        note = "CHECK: far from configured coordinates" if r.suspicious else r.source
        lines.append(f"{city.id:<20} {r.location:<45} {reach:>12} {km:>7}  {note}")
    return "\n".join(lines)


# --------------------------------------------------------------------- plan


@dataclass(frozen=True)
class Cell:
    """One (query, city) search in the collection grid."""

    query: Query
    city: City


def plan_names(settings: Settings) -> list[str]:
    """Return the plans ``collect`` accepts: those in settings.yaml, plus ``grid``."""
    return [*settings.plans, "grid"]


def plan_cells(settings: Settings, plan: str) -> list[Cell]:
    """Return the cells of a plan in query-major order.

    Query-major means all cities for one query are fetched back to back, so content
    that changes over time (scores, news) is held roughly constant across cities.
    """
    if plan == "grid":
        queries, cities = list(settings.queries), list(settings.cities)
    elif plan in settings.plans:
        # Keep config order (not the plan's listing order) so plans line up with the grid.
        wanted = settings.plans[plan]
        queries = [q for q in settings.queries if q.id in wanted.queries]
        cities = [c for c in settings.cities if c.id in wanted.cities]
    else:
        raise ValueError(f"Unknown plan {plan!r}; expected one of {plan_names(settings)}")
    return [Cell(q, c) for q in queries for c in cities]


@dataclass
class Budget:
    """How many paid searches a run may make, and why."""

    spent: int
    limit: int
    account_left: int | None
    planned_new: int

    @property
    def allowed(self) -> int:
        """Return the number of new paid searches this run may make."""
        local = max(0, self.limit - self.spent)
        return local if self.account_left is None else min(local, self.account_left)

    @property
    def fits(self) -> bool:
        """Return whether all planned searches fit in the budget."""
        return self.planned_new <= self.allowed


def account_searches_left(client: SerpApiClient) -> int | None:
    """Return searches left on the SerpAPI account this month, if the API reports it."""
    acct = client.account()
    for key in ("total_searches_left", "plan_searches_left"):
        if isinstance(acct.get(key), int):
            return int(acct[key])
    return None


def describe_plan(
    client: SerpApiClient,
    cells: list[Cell],
    locations: dict[str, ResolvedLocation],
    account_left: int | None,
    limit: int | None = None,
) -> tuple[str, Budget, list[Cell]]:
    """Describe what a run would do without making any requests.

    Returns:
        The printable plan, the budget, and the cells that would be fetched (after
        ``limit``).
    """
    new = [c for c in cells if not client.is_cached(c.query.id, c.city.id)]
    to_fetch = new if limit is None else new[: max(0, limit)]
    budget = Budget(
        spent=client.ledger.spent(),
        limit=client.budget_limit,
        account_left=account_left,
        planned_new=len(to_fetch),
    )
    fetch_ids = {(c.query.id, c.city.id) for c in to_fetch}
    lines = [f"{'#':>3}  {'query':<28} {'city':<20} {'location':<42} status"]
    for i, c in enumerate(cells, 1):
        loc = locations.get(c.city.id)
        if client.is_cached(c.query.id, c.city.id):
            status = "cached (free)"
        elif (c.query.id, c.city.id) in fetch_ids:
            status = "NEW (1 search)"
        else:
            status = "skipped (--limit)"
        loc_name = loc.location if loc else "<unresolved: run `serpint locations`>"
        lines.append(f"{i:>3}  {c.query.id:<28} {c.city.id:<20} {loc_name:<42} {status}")
    acct = "not checked" if account_left is None else str(account_left)
    lines += [
        "",
        f"cells in plan:            {len(cells)}",
        f"already cached:           {len(cells) - len(new)}",
        f"new paid searches:        {budget.planned_new}",
        f"ledger spend so far:      {budget.spent}",
        f"ledger spend after run:   {budget.spent + budget.planned_new} / {budget.limit}",
        f"account searches left:    {acct}",
        f"fits budget:              {'yes' if budget.fits else 'NO'}",
    ]
    return "\n".join(lines), budget, to_fetch


@dataclass
class RunReport:
    """Outcome of a collection run."""

    fetched: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    html_failed: dict[str, str] = field(default_factory=dict)


def run(
    client: SerpApiClient,
    cells: list[Cell],
    locations: dict[str, ResolvedLocation],
    to_fetch: list[Cell],
    echo: Printer = print,
) -> RunReport:
    """Fetch the planned cells, then make sure every cached cell has its raw HTML.

    Resumable: cached cells are skipped, so re-running after an interruption picks up
    where it stopped. A failed search is reported and not retried within the run.
    """
    report = RunReport()
    fetch_ids = {(c.query.id, c.city.id) for c in to_fetch}
    with client.lock():
        for c in cells:
            tag = f"{c.query.id}@{c.city.id}"
            if client.is_cached(c.query.id, c.city.id):
                report.cached.append(tag)
            elif (c.query.id, c.city.id) in fetch_ids:
                try:
                    client.search(c.query, c.city, locations[c.city.id].location)
                except BudgetExceededError as e:
                    report.failed[tag] = str(e)
                    echo(f"STOP {tag}: {e}")
                    break
                except SerpApiError as e:
                    report.failed[tag] = str(e)
                    echo(f"FAIL {tag}: {e}")
                    continue
                report.fetched.append(tag)
                echo(f"ok   {tag}  (ledger: {client.ledger.spent()})")
            else:
                continue
            hit = client.cached(c.query.id, c.city.id)
            if hit is not None:
                try:
                    client.fetch_raw_html(hit, c.query.id, c.city.id)
                except SerpApiError as e:
                    report.html_failed[tag] = str(e)
                    echo(f"warn {tag}: raw HTML not saved ({e})")
    return report


# ---------------------------------------------------------------- inventory


def _type_name(v: Any) -> str:
    return "null" if v is None else type(v).__name__


def field_inventory(
    responses: Iterable[tuple[str, str, dict[str, Any]]], settings: Settings
) -> str:
    """Summarize which fields SerpAPI returned, as Markdown.

    Args:
        responses: ``(query_id, city_id, response_json)`` triples.
        settings: Used for query tiers and display order.

    Returns:
        A Markdown report: top-level keys with frequency, type and size; sub-keys of each
        element; and a query × element presence matrix.
    """
    rows = list(responses)
    n = len(rows)
    presence: Counter[str] = Counter()
    types: dict[str, Counter[str]] = defaultdict(Counter)
    sizes: dict[str, list[int]] = defaultdict(list)
    subkeys: dict[str, Counter[str]] = defaultdict(Counter)
    subkey_den: Counter[str] = Counter()
    by_query: dict[str, Counter[str]] = defaultdict(Counter)
    per_query_n: Counter[str] = Counter()

    for qid, _cid, data in rows:
        per_query_n[qid] += 1
        for key, val in data.items():
            presence[key] += 1
            by_query[qid][key] += 1
            types[key][_type_name(val)] += 1
            if isinstance(val, list):
                sizes[key].append(len(val))
                for item in val:
                    if isinstance(item, dict):
                        subkey_den[key] += 1
                        subkeys[key].update(item.keys())
            elif isinstance(val, dict):
                sizes[key].append(len(val))
                subkey_den[key] += 1
                subkeys[key].update(val.keys())

    order = sorted(presence, key=lambda k: (-presence[k], k))
    out = [
        f"# Field inventory ({n} responses)",
        "",
        "Generated by `serpint inspect` from cached SerpAPI responses.",
        "",
        "## Top-level keys",
        "",
        "| key | present | type | size (median / max) |",
        "|---|---|---|---|",
    ]
    for k in order:
        t = ", ".join(sorted(types[k]))
        sz = sizes.get(k)
        size = f"{statistics.median(sz):g} / {max(sz)}" if sz else ""
        out.append(f"| `{k}` | {presence[k]}/{n} | {t} | {size} |")

    out += ["", "## Sub-keys", "", "Share of items (list) or objects (dict) carrying each key.", ""]
    for k in order:
        if not subkeys.get(k):
            continue
        den = subkey_den[k]
        keys = ", ".join(
            f"`{sk}` {cnt / den:.0%}"
            for sk, cnt in sorted(subkeys[k].items(), key=lambda kv: (-kv[1], kv[0]))
        )
        out += [f"- **`{k}`**: {keys}"]

    queries = [q for q in settings.queries if q.id in per_query_n]
    out += [
        "",
        "## Element presence by query",
        "",
        "Cities (out of those fetched) whose page had each element.",
        "",
        "| key | " + " | ".join(f"{q.id} ({q.tier})" for q in queries) + " |",
        "|---|" + "---|" * len(queries),
    ]
    for k in order:
        cells = [f"{by_query[q.id][k]}/{per_query_n[q.id]}" for q in queries]
        out.append(f"| `{k}` | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"
