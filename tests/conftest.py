"""Shared fixtures: settings pointed at a temp data dir, and a recording fake HTTP session."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from typing import Any

import pytest
import requests

from serpint.config import Settings, load_settings

SECRET = "test-secret-key-0123456789"


def make_response(payload: Any, status: int = 200, raw: bytes | None = None) -> requests.Response:
    r = requests.Response()
    r.status_code = status
    r._content = raw if raw is not None else json.dumps(payload).encode()
    return r


class FakeSession:
    """Records every GET and answers with a handler(url, params) -> Response."""

    def __init__(self, handler: Callable[[str, dict[str, Any]], requests.Response]) -> None:
        self.handler = handler
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, params: dict[str, Any], timeout: float) -> requests.Response:
        self.calls.append((url, dict(params)))
        return self.handler(url, params)

    def calls_to(self, fragment: str) -> list[tuple[str, dict[str, Any]]]:
        return [c for c in self.calls if fragment in c[0]]


def ok_search(url: str, params: dict[str, Any]) -> requests.Response:
    """A minimal successful search payload (shape only; real fields come from the pilot)."""
    if "search.json" in url:
        sid = f"{params['q']}-{params['location']}".replace(" ", "_")
        return make_response(
            {
                "search_metadata": {
                    "id": sid,
                    "status": "Success",
                    "raw_html_file": f"https://serpapi.com/searches/{sid}/{sid}.html",
                },
                "search_parameters": {k: v for k, v in params.items() if k != "api_key"},
                "organic_results": [{"position": 1, "link": "https://example.org/"}],
            }
        )
    if url.endswith(".html"):
        return make_response(None, raw=b"<html><body>results</body></html>")
    if "account.json" in url:
        return make_response({"total_searches_left": 1000})
    raise AssertionError(f"unexpected URL {url}")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return dataclasses.replace(load_settings(), data_dir=tmp_path / "data")


@pytest.fixture
def session() -> FakeSession:
    return FakeSession(ok_search)
