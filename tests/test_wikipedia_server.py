from __future__ import annotations

import unittest
from datetime import date
from typing import Any

import requests

from mcp_server.wikipedia_server import (
    collect_trending_articles,
    is_commercially_usable_title,
    normalize_article_title,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = iter(responses)
        self.headers: dict[str, str] = {}
        self.urls: list[str] = []

    def get(self, url: str, timeout: tuple[int, int]) -> FakeResponse:
        self.urls.append(url)
        self.timeout = timeout
        return next(self.responses)


class WikipediaServerTests(unittest.TestCase):
    def test_normalizes_and_filters_namespace_noise(self) -> None:
        self.assertEqual(normalize_article_title("Solar_Panels%20Guide"), "Solar Panels Guide")
        self.assertFalse(is_commercially_usable_title("Main Page"))
        self.assertFalse(is_commercially_usable_title("Special:Search"))
        self.assertFalse(is_commercially_usable_title("Wikipedia:About"))
        self.assertTrue(is_commercially_usable_title("Electric bicycle"))

    def test_aggregates_processed_days_and_skips_404(self) -> None:
        payload_one = {
            "items": [
                {
                    "articles": [
                        {"article": "Heat_pump", "views": 6000},
                        {"article": "Main_Page", "views": 999999},
                    ]
                }
            ]
        }
        payload_two = {
            "items": [
                {
                    "articles": [
                        {"article": "Heat_pump", "views": 7000},
                        {"article": "Electric_bicycle", "views": 11000},
                    ]
                }
            ]
        }
        session = FakeSession(
            [FakeResponse(200, payload_one), FakeResponse(404), FakeResponse(200, payload_two)]
        )

        result = collect_trending_articles(
            days=3,
            limit=10,
            min_views=10_000,
            today=date(2026, 7, 20),
            session=session,  # type: ignore[arg-type]
        )

        self.assertEqual(
            result,
            [
                {"title": "Heat pump", "views": 13000},
                {"title": "Electric bicycle", "views": 11000},
            ],
        )
        self.assertTrue(session.urls[0].endswith("/2026/07/17"))
        self.assertTrue(session.urls[-1].endswith("/2026/07/15"))

    def test_raises_if_every_day_is_missing(self) -> None:
        session = FakeSession([FakeResponse(404), FakeResponse(404)])
        with self.assertRaisesRegex(RuntimeError, "no usable processed days"):
            collect_trending_articles(
                days=2,
                today=date(2026, 7, 20),
                session=session,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
