"""Typed loading of the YAML files in ``config/``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TIERS = ("explicit_local", "implicit_local", "neutral")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class City:
    """A city we simulate searches from."""

    id: str
    name: str
    state: str
    lat: float
    lon: float
    aliases: tuple[str, ...]
    location: str | None = None  # manual override of the SerpAPI canonical location


@dataclass(frozen=True)
class Query:
    """A search query and its locality tier."""

    id: str
    text: str
    tier: str


@dataclass(frozen=True)
class Settings:
    """Everything in ``config/``, validated."""

    search_params: dict[str, str]
    max_searches: int
    check_account: bool
    data_dir: Path
    geoguessr_scale_km: float
    seed: int
    cities: tuple[City, ...]
    queries: tuple[Query, ...]
    pilot_cities: tuple[str, ...]
    pilot_queries: tuple[str, ...]
    root: Path = field(default=PROJECT_ROOT)

    def city(self, city_id: str) -> City:
        """Return the city with this id."""
        return {c.id: c for c in self.cities}[city_id]

    def query(self, query_id: str) -> Query:
        """Return the query with this id."""
        return {q.id: q for q in self.queries}[query_id]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _unique(ids: list[str], what: str) -> None:
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate {what} ids: {sorted(dupes)}")


def load_settings(config_dir: Path | None = None, root: Path | None = None) -> Settings:
    """Load and validate ``settings.yaml``, ``cities.yaml`` and ``queries.yaml``.

    Args:
        config_dir: Directory holding the YAML files. Defaults to ``<root>/config``.
        root: Project root that relative paths (e.g. ``data_dir``) resolve against.

    Returns:
        The validated settings.

    Raises:
        ValueError: If ids are duplicated, a tier is unknown, or the pilot names an
            unknown city or query.
    """
    root = root or PROJECT_ROOT
    config_dir = config_dir or root / "config"
    s = _read_yaml(config_dir / "settings.yaml")
    cities = tuple(
        City(
            id=c["id"],
            name=c["name"],
            state=c["state"],
            lat=float(c["lat"]),
            lon=float(c["lon"]),
            aliases=tuple(c.get("aliases") or [c["name"]]),
            location=c.get("location"),
        )
        for c in _read_yaml(config_dir / "cities.yaml")["cities"]
    )
    queries = tuple(
        Query(id=q["id"], text=q["text"], tier=q["tier"])
        for q in _read_yaml(config_dir / "queries.yaml")["queries"]
    )
    _unique([c.id for c in cities], "city")
    _unique([q.id for q in queries], "query")
    bad_tiers = {q.tier for q in queries} - set(TIERS)
    if bad_tiers:
        raise ValueError(f"Unknown query tiers: {sorted(bad_tiers)}; expected one of {TIERS}")

    pilot = s.get("pilot", {})
    pilot_cities = tuple(pilot.get("cities", []))
    pilot_queries = tuple(pilot.get("queries", []))
    unknown = (set(pilot_cities) - {c.id for c in cities}) | (
        set(pilot_queries) - {q.id for q in queries}
    )
    if unknown:
        raise ValueError(f"Pilot names unknown ids: {sorted(unknown)}")

    data_dir = Path(s["paths"]["data_dir"])
    return Settings(
        search_params={k: str(v) for k, v in s["search"].items()},
        max_searches=int(s["budget"]["max_searches"]),
        check_account=bool(s["budget"].get("check_account", True)),
        data_dir=data_dir if data_dir.is_absolute() else root / data_dir,
        geoguessr_scale_km=float(s["scoring"]["geoguessr_scale_km"]),
        seed=int(s["seed"]),
        cities=cities,
        queries=queries,
        pilot_cities=pilot_cities,
        pilot_queries=pilot_queries,
        root=root,
    )
