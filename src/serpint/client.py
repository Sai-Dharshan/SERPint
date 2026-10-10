"""The only path to SerpAPI: disk cache, call ledger, budget guard and key hygiene.

Every paid Google search goes through :meth:`SerpApiClient.search`, which

* returns the cached response from ``data/raw/{query_id}/{city_id}/{timestamp}.json`` if
  one exists, and never re-fetches it;
* otherwise checks the ledger against the budget, records the attempt in
  ``data/ledger.jsonl`` *before* the request goes out (so a crash can only over-count
  spend, never under-count it), makes the call and caches the response atomically.

The free endpoints (Locations API, Account API, the ``raw_html_file`` archive) do not
touch the ledger. The API key is only ever sent as a request parameter and is redacted
from every error message; a response is refused if it would write the key to disk.
"""

from __future__ import annotations

import contextlib
import fcntl
import gzip
import json
import os
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import requests

from serpint.config import City, Query, Settings

SEARCH_URL = "https://serpapi.com/search.json"
LOCATIONS_URL = "https://serpapi.com/locations.json"
ACCOUNT_URL = "https://serpapi.com/account.json"
TIMEOUT_S = 90
# SerpAPI returns this as an "error" for a search that ran fine but found nothing. It is a
# real observation of the page, so it is cached like any other response.
NO_RESULTS_ERROR = "Google hasn't returned any results for this query."


class SerpApiError(RuntimeError):
    """A SerpAPI request failed. The message never contains the API key."""


class BudgetExceededError(RuntimeError):
    """A paid search would take the ledger past the allowed number of searches."""


class HttpSession(Protocol):
    """The part of :class:`requests.Session` the client uses (lets tests inject a fake)."""

    def get(self, url: str, params: dict[str, Any], timeout: float) -> requests.Response:
        """Send a GET request."""
        ...


def utc_timestamp() -> str:
    """Return the current UTC time as a filename-safe ``YYYYmmddTHHMMSSffffffZ`` string."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def redact(text: str, secret: str | None) -> str:
    """Replace every occurrence of ``secret`` (and any ``api_key=...`` value) with ``***``."""
    if secret:
        text = text.replace(secret, "***")
    return re.sub(r"(api_key=)[^&\s'\"]+", r"\1***", text)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file + rename, so readers never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


class Ledger:
    """Append-only JSONL record of every paid search attempt.

    Each paid call writes a ``started`` line before the request and a ``finished`` line
    after it. Spend is the number of ``started`` lines, so interrupted calls count.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def entries(self) -> list[dict[str, Any]]:
        """Return all ledger lines, oldest first."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def spent(self) -> int:
        """Return the number of paid search attempts recorded so far."""
        return sum(1 for e in self.entries() if e.get("event") == "started")

    def append(self, entry: dict[str, Any]) -> None:
        """Append one line and fsync it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())


@dataclass(frozen=True)
class CachedResponse:
    """A cached search response and where it lives on disk."""

    path: Path
    timestamp: str
    data: dict[str, Any]


class SerpApiClient:
    """Budget-guarded, caching SerpAPI client.

    Args:
        settings: Project settings (search parameters, budget, data directory).
        api_key: SerpAPI key. Needed only for calls that are not served from cache.
        allow_over_budget: Extra paid searches beyond ``settings.max_searches`` that the
            user has explicitly approved for this run.
        session: HTTP session; tests pass a fake.
    """

    def __init__(
        self,
        settings: Settings,
        api_key: str | None,
        allow_over_budget: int = 0,
        session: HttpSession | None = None,
    ) -> None:
        self.settings = settings
        self._api_key = api_key or None
        self.allow_over_budget = max(0, allow_over_budget)
        self.session: HttpSession = session or requests.Session()
        self.data_dir = settings.data_dir
        self.raw_dir = self.data_dir / "raw"
        self.html_dir = self.data_dir / "html"
        self.locations_raw_dir = self.data_dir / "locations_raw"
        self.ledger = Ledger(self.data_dir / "ledger.jsonl")

    # ----------------------------------------------------------------- budget

    @property
    def budget_limit(self) -> int:
        """Return the most paid searches the ledger may reach in this run."""
        return self.settings.max_searches + self.allow_over_budget

    def remaining_budget(self) -> int:
        """Return how many more paid searches the local budget allows."""
        return max(0, self.budget_limit - self.ledger.spent())

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Hold an exclusive lock on the data directory so two runs can't overlap.

        Raises:
            RuntimeError: If another process holds the lock.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with (self.data_dir / ".lock").open("w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                raise RuntimeError("Another serpint run holds data/.lock") from e
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    # ------------------------------------------------------------------ cache

    def cell_dir(self, query_id: str, city_id: str) -> Path:
        """Return the cache directory for one (query, city) cell."""
        return self.raw_dir / query_id / city_id

    def cached(self, query_id: str, city_id: str) -> CachedResponse | None:
        """Return the latest cached response for a cell, or ``None`` if there is none."""
        files = sorted(self.cell_dir(query_id, city_id).glob("*.json"))
        if not files:
            return None
        path = files[-1]
        with path.open(encoding="utf-8") as f:
            return CachedResponse(path=path, timestamp=path.stem, data=json.load(f))

    def is_cached(self, query_id: str, city_id: str) -> bool:
        """Return whether a cell already has a cached response."""
        return any(self.cell_dir(query_id, city_id).glob("*.json"))

    def html_path(self, query_id: str, city_id: str, timestamp: str) -> Path:
        """Return where the raw HTML for a cached response is (or would be) stored."""
        return self.html_dir / query_id / city_id / f"{timestamp}.html.gz"

    # ------------------------------------------------------------------- HTTP

    def _require_key(self) -> str:
        if not self._api_key:
            raise SerpApiError(
                "SERPAPI_API_KEY is not set. Put it in .env (see .env.example) or the environment."
            )
        return self._api_key

    def _get(self, url: str, params: dict[str, Any]) -> requests.Response:
        """GET with the key redacted from any exception text."""
        try:
            return self.session.get(url, params=params, timeout=TIMEOUT_S)
        except requests.RequestException as e:
            raise SerpApiError(redact(f"{type(e).__name__}: {e}", self._api_key)) from None

    def _json_or_error(self, resp: requests.Response, what: str) -> Any:
        try:
            data = resp.json()
        except ValueError:
            data = None
        if resp.status_code != 200:
            detail = data.get("error") if isinstance(data, dict) else resp.text[:300]
            raise SerpApiError(redact(f"{what}: HTTP {resp.status_code}: {detail}", self._api_key))
        if data is None:
            raise SerpApiError(f"{what}: response was not JSON")
        return data

    def _dump_without_key(self, data: Any) -> bytes:
        text = json.dumps(data, ensure_ascii=False, indent=1)
        if self._api_key and self._api_key in text:
            raise SerpApiError("Refusing to write a response that contains the API key")
        return text.encode("utf-8")

    # ----------------------------------------------------------- paid search

    def search(self, query: Query, city: City, location: str) -> CachedResponse:
        """Return the Google results for ``query`` searched from ``location``.

        Served from cache when possible; otherwise one paid search is made.

        Args:
            query: The query to search.
            city: The city the search is attributed to (used for the cache path).
            location: SerpAPI canonical location name for ``city``.

        Returns:
            The cached (possibly just-fetched) response.

        Raises:
            BudgetExceededError: If the call would exceed the allowed budget.
            SerpApiError: If the request fails or SerpAPI reports an error.
        """
        hit = self.cached(query.id, city.id)
        if hit is not None:
            return hit

        key = self._require_key()
        spent = self.ledger.spent()
        if spent + 1 > self.budget_limit:
            raise BudgetExceededError(
                f"Ledger shows {spent} paid searches; one more would exceed the limit of "
                f"{self.budget_limit} (max_searches={self.settings.max_searches}, "
                f"allow_over_budget={self.allow_over_budget})."
            )

        params = {**self.settings.search_params, "q": query.text, "location": location}
        call_id = uuid.uuid4().hex
        ts = utc_timestamp()
        self.ledger.append(
            {
                "event": "started",
                "call_id": call_id,
                "ts": ts,
                "query_id": query.id,
                "city_id": city.id,
                "params": params,
            }
        )
        try:
            resp = self._get(SEARCH_URL, {**params, "api_key": key})
            data = resp.json() if resp.content else {}
        except (SerpApiError, ValueError) as e:
            self._finish(call_id, "error", error=redact(str(e), key))
            raise SerpApiError(redact(str(e), key)) from None

        if not isinstance(data, dict):
            data = {"error": "response was not a JSON object"}
        error = data.get("error")
        if resp.status_code != 200 or (error and error != NO_RESULTS_ERROR):
            msg = redact(f"HTTP {resp.status_code}: {error}", key)
            self._finish(call_id, "error", error=msg)
            raise SerpApiError(f"Search {query.id}@{city.id} failed: {msg}")

        path = self.cell_dir(query.id, city.id) / f"{ts}.json"
        atomic_write_bytes(path, self._dump_without_key(data))
        self._finish(
            call_id,
            "no_results" if error else "ok",
            search_id=(data.get("search_metadata") or {}).get("id"),
            path=str(path.relative_to(self.data_dir)),
        )
        return CachedResponse(path=path, timestamp=ts, data=data)

    def _finish(self, call_id: str, status: str, **extra: Any) -> None:
        self.ledger.append(
            {"event": "finished", "call_id": call_id, "ts": utc_timestamp(), "status": status}
            | extra
        )

    # ------------------------------------------------------- free endpoints

    def fetch_raw_html(self, cached: CachedResponse, query_id: str, city_id: str) -> Path | None:
        """Download and gzip the ``raw_html_file`` of a cached response, if not already done.

        Free: SerpAPI's search archive does not consume search credits.

        Returns:
            The path of the gzipped HTML, or ``None`` if the response has no raw HTML link.
        """
        out = self.html_path(query_id, city_id, cached.timestamp)
        if out.exists():
            return out
        url = (cached.data.get("search_metadata") or {}).get("raw_html_file")
        if not url:
            return None
        resp = self._get(url, {"api_key": self._require_key()})
        if resp.status_code != 200:
            raise SerpApiError(f"raw_html_file for {query_id}@{city_id}: HTTP {resp.status_code}")
        atomic_write_bytes(out, gzip.compress(resp.content))
        return out

    def locations(self, q: str, limit: int = 10) -> list[dict[str, Any]]:
        """Return SerpAPI Locations API candidates for ``q`` (cached on disk; free)."""
        slug = re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")
        path = self.locations_raw_dir / f"{slug}.json"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return json.load(f)
        data = self._json_or_error(
            self._get(LOCATIONS_URL, {"q": q, "limit": limit}), f"Locations API ({q})"
        )
        if not isinstance(data, list):
            raise SerpApiError(f"Locations API ({q}): expected a list, got {type(data).__name__}")
        atomic_write_bytes(path, self._dump_without_key(data))
        return data

    def account(self) -> dict[str, Any]:
        """Return the SerpAPI Account API response (free; not cached)."""
        data = self._json_or_error(
            self._get(ACCOUNT_URL, {"api_key": self._require_key()}), "Account API"
        )
        if not isinstance(data, dict):
            raise SerpApiError("Account API: expected a JSON object")
        return data
