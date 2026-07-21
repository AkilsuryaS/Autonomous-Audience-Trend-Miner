"""FastMCP wrapper around Wikimedia's public Pageviews API.

This module deliberately contains no LLM or presentation logic. The agent layer
can only reach Wikimedia through the MCP tool exposed here.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, timedelta
from typing import Any
from urllib.parse import unquote

import requests
from mcp.server.fastmcp import FastMCP

LOGGER = logging.getLogger("wikipedia_mcp")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

WIKIMEDIA_TOP_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
    "en.wikipedia/all-access/{year:04d}/{month:02d}/{day:02d}"
)
USER_AGENT = (
    "InMarket-Audience-Trend-Miner/1.0 "
    "(educational prototype; Wikimedia Pageviews API)"
)
EXCLUDED_TITLES = {"main page"}
EXCLUDED_PREFIXES = (
    "wikipedia:",
    "special:",
    "portal:",
    "category:",
    "file:",
)

mcp = FastMCP("Wikipedia Trending Data Service")


def normalize_article_title(raw_title: str) -> str:
    """Convert API-style titles into readable, mergeable article titles."""

    return " ".join(unquote(raw_title).replace("_", " ").split())


def is_commercially_usable_title(title: str) -> bool:
    """Remove Wikimedia namespaces and known utility-page noise."""

    folded = title.casefold()
    return bool(title) and folded not in EXCLUDED_TITLES and not folded.startswith(
        EXCLUDED_PREFIXES
    )


def collect_trending_articles(
    *,
    days: int = 7,
    limit: int = 100,
    min_views: int = 10_000,
    today: date | None = None,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch, normalize, aggregate, filter, and rank processed daily pageviews.

    The newest requested date is three days before ``today`` because Wikimedia
    pageview aggregation commonly trails real time by one or two days. A failed
    day is logged and skipped; the run only fails when every requested day is
    unavailable.
    """

    if not 1 <= days <= 14:
        raise ValueError("days must be between 1 and 14")
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    if min_views < 0:
        raise ValueError("min_views cannot be negative")

    request_session = session or requests.Session()
    owns_session = session is None
    request_session.headers.update({"User-Agent": USER_AGENT})
    anchor = (today or date.today()) - timedelta(days=3)
    totals: defaultdict[str, int] = defaultdict(int)
    successful_days = 0

    try:
        for offset in range(days):
            target = anchor - timedelta(days=offset)
            url = WIKIMEDIA_TOP_URL.format(
                year=target.year,
                month=target.month,
                day=target.day,
            )
            try:
                response = request_session.get(url, timeout=(5, 20))
                if response.status_code == 404:
                    LOGGER.warning("Skipping unprocessed Wikimedia day %s (404)", target)
                    continue
                response.raise_for_status()
                payload = response.json()
                daily_articles = payload.get("items", [{}])[0].get("articles", [])
                successful_days += 1
            except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                LOGGER.warning("Skipping Wikimedia day %s: %s", target, exc)
                continue

            for article in daily_articles:
                try:
                    title = normalize_article_title(str(article["article"]))
                    views = int(article["views"])
                except (KeyError, TypeError, ValueError):
                    LOGGER.debug("Ignoring malformed article row: %r", article)
                    continue
                if is_commercially_usable_title(title):
                    totals[title] += views
    finally:
        if owns_session:
            request_session.close()

    if successful_days == 0:
        raise RuntimeError("Wikimedia returned no usable processed days")

    ranked = [
        {"title": title, "views": views}
        for title, views in totals.items()
        if views >= min_views
    ]
    ranked.sort(key=lambda article: (-article["views"], article["title"]))
    LOGGER.info(
        "Aggregated %s articles from %s/%s processed days; returning top %s",
        len(ranked),
        successful_days,
        days,
        min(limit, len(ranked)),
    )
    return ranked[:limit]


@mcp.tool()
def fetch_trending_wikipedia_articles(
    days: int = 7,
    limit: int = 100,
    min_views: int = 10_000,
) -> list[dict[str, Any]]:
    """Return globally trending English Wikipedia articles.

    Results aggregate the latest processed ``days`` ending three days before
    today. They are global English-Wikipedia readership signals, not US traffic.
    Utility namespaces and low-view noise are removed before ranking.
    """

    return collect_trending_articles(days=days, limit=limit, min_views=min_views)


if __name__ == "__main__":
    # stdio is explicit so the LangChain client can spawn this process on demand.
    mcp.run(transport="stdio")
