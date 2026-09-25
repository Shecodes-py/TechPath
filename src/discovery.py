import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from apify_client import ApifyClient

logger = logging.getLogger("TechPath.Discovery")

GOOGLE_SEARCH_ACTOR = "apify/google-search-scraper"
YOUTUBE_SCRAPER_ACTOR = "streamers/youtube-scraper"

_PLACEHOLDER_TOKENS = {"", "your_apify_token_here"}


class ResourceDiscovery:
    """
    Web & opportunity discovery via Apify Store Actors (Google Search + YouTube).
    Falls back to an empty result set when no usable token / Actor context exists.
    """

    def __init__(self, apify_token: Optional[str] = None):
        raw = apify_token or os.getenv("APIFY_TOKEN")
        self.apify_token = None if not raw or raw in _PLACEHOLDER_TOKENS else raw
        self.client = ApifyClient(self.apify_token) if self.apify_token else None

    async def discover(self, goal: str, location: str, interests: List[str], current_level: str = "Beginner") -> Dict[str, Any]:
        level_str = (current_level or "Beginner").lower()
        level_suffix = "advanced architecture" if level_str == "advanced" else ("intermediate system design" if level_str == "intermediate" else "beginner fundamentals")
        
        video_query = f"{goal} tutorial {level_suffix} {' '.join(interests[:2])}".strip()
        queries = [
            f"best free {level_suffix} courses for {goal}",
            f"{goal} prerequisites learning roadmap",
            f"hackathons in {location} or remote tech opportunities 2026",
            f"top {goal} projects {' '.join(interests)}",
        ]

        results: Dict[str, Any] = {
            "queries_executed": queries,
            "search_results": [],
            "video_results": [],
            "discovery_mode": "none",
        }

        if not self.client and not _actor_is_ready():
            results["discovery_mode"] = "no_token"
            logger.info("No usable Apify token / Actor context; skipping live scrape.")
            return results

        google_task = self._call_google_search(queries[:3])
        youtube_task = self._call_youtube_search(video_query)
        google_raw, youtube_raw = await asyncio.gather(
            google_task, youtube_task, return_exceptions=True
        )

        if isinstance(google_raw, Exception):
            logger.warning(f"Google Search Actor failed: {google_raw}")
            google_raw = []
        if isinstance(youtube_raw, Exception):
            logger.warning(f"YouTube Scraper Actor failed: {youtube_raw}")
            youtube_raw = []

        results["search_results"] = _flatten_google_items(google_raw)[:15]
        results["video_results"] = _flatten_youtube_items(youtube_raw)[:10]
        if results["search_results"] or results["video_results"]:
            results["discovery_mode"] = "apify_store_actors"
        elif self.client or _actor_is_ready():
            results["discovery_mode"] = "apify_empty"
        else:
            results["discovery_mode"] = "no_token"
            logger.info("No usable Apify token / Actor context; skipping live scrape.")

        logger.info(
            f"Discovery complete: {len(results['search_results'])} search hits, "
            f"{len(results['video_results'])} videos ({results['discovery_mode']})."
        )
        return results

    async def _call_google_search(self, queries: List[str] | str, max_results: int = 10) -> List[Dict[str, Any]]:
        query_input = "\n".join(queries) if isinstance(queries, list) else queries
        logger.info(f"Calling Google Search Scraper for queries: {queries!r}")
        return await self._run_actor_and_get_items(
            GOOGLE_SEARCH_ACTOR,
            {
                "queries": query_input,
                "maxPagesPerQuery": 1,
                "resultsPerPage": max_results,
            },
        )

    async def _call_youtube_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Calling YouTube Scraper for: {query!r}")
        return await self._run_actor_and_get_items(
            YOUTUBE_SCRAPER_ACTOR,
            {
                "searchQueries": [query],
                "maxResults": max_results,
            },
        )

    async def _run_actor_and_get_items(
        self, actor_id: str, run_input: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        actor_items = await self._run_via_actor_sdk(actor_id, run_input)
        if actor_items is not None:
            return actor_items
        if not self.client:
            return []
        return await asyncio.to_thread(self._run_via_sync_client, actor_id, run_input)

    async def _run_via_actor_sdk(
        self, actor_id: str, run_input: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        try:
            from apify import Actor

            if not hasattr(Actor, "call"):
                return None
            run = await Actor.call(actor_id, run_input=run_input)
            if run is None:
                return []
            run_id = run.get("id") if isinstance(run, dict) else getattr(run, "id", None)
            if not run_id:
                return []
            listed = await Actor.apify_client.run(run_id).dataset().list_items()
            items = getattr(listed, "items", None)
            if items is None and isinstance(listed, dict):
                items = listed.get("items", [])
            logger.info(f"{actor_id} returned {len(items)} items")
            return list(items or [])
        except Exception as e:
            logger.debug(f"Actor.call for {actor_id} unavailable: {e}")
            return None

    def _run_via_sync_client(self, actor_id: str, run_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        run = self.client.actor(actor_id).call(run_input=run_input)
        dataset_id = getattr(run, "default_dataset_id", None) or (
            run.get("defaultDatasetId") if isinstance(run, dict) else None
        )
        if not dataset_id:
            return []
        items = list(self.client.dataset(dataset_id).iterate_items())
        logger.info(f"{actor_id} returned {len(items)} items via ApifyClient")
        return items


def _actor_is_ready() -> bool:
    try:
        from apify import Actor

        for name in ("is_initialized", "_is_initialized"):
            initialized = getattr(Actor, name, None)
            if callable(initialized):
                if initialized():
                    return True
            elif initialized:
                return True
        return False
    except Exception:
        return False


def _flatten_google_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for item in items or []:
        organics = item.get("organicResults") or item.get("organic_results") or []
        rows = organics if organics else [item]
        for row in rows:
            title = row.get("title")
            url = row.get("url") or row.get("link")
            if not title or not url:
                continue
            flattened.append(
                {
                    "title": title,
                    "url": url,
                    "description": row.get("description") or row.get("snippet") or "",
                    "source": "google",
                }
            )
    return flattened


def _flatten_youtube_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for item in items or []:
        title = item.get("title")
        url = item.get("url") or item.get("id")
        if url and not str(url).startswith("http"):
            url = f"https://www.youtube.com/watch?v={url}"
        if not title or not url:
            continue
        flattened.append(
            {
                "title": title,
                "url": url,
                "channelName": item.get("channelName") or item.get("channel") or "",
                "source": "youtube",
            }
        )
    return flattened
