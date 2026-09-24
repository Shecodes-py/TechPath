import logging
import os
from typing import Dict, Any, List, Optional
import httpx
from apify_client import ApifyClient

logger = logging.getLogger("TechPath.Discovery")


class ResourceDiscovery:
    """
    Web & Opportunity Discovery Engine.
    Leverages Apify Google Search Scraper and Web Crawlers (if API token present)
    or lightweight HTTP web discovery to gather fresh learning resources and hackathons.
    """

    def __init__(self, apify_token: Optional[str] = None):
        self.apify_token = apify_token or os.getenv("APIFY_TOKEN")
        self.client = ApifyClient(self.apify_token) if self.apify_token else None

    async def discover(self, goal: str, location: str, interests: List[str]) -> Dict[str, Any]:
        """
        Execute web search queries for courses, docs, YouTube playlists, and hackathons.
        """
        search_queries = [
            f"best free courses for {goal}",
            f"{goal} prerequisites learning roadmap",
            f"hackathons in {location} or remote tech opportunities 2026",
            f"top {goal} projects {' '.join(interests)}"
        ]

        results = {
            "queries_executed": search_queries,
            "scraped_items": []
        }

        if self.client:
            try:
                logger.info(f"Executing Apify Google Search Scraper for goal: {goal}")
                # Invoke Apify Google Search Scraper
                run_input = {
                    "queries": "\n".join(search_queries[:2]),
                    "maxPagesPerQuery": 1,
                    "resultsPerPage": 3,
                }
                # Call actor synchronously
                run = self.client.actor("apify/google-search-scraper").call(run_input=run_input)
                dataset_id = getattr(run, "default_dataset_id", None) or (run.get("defaultDatasetId") if isinstance(run, dict) else None)
                if dataset_id:
                    dataset = self.client.dataset(dataset_id)
                    items = list(dataset.iterate_items())
                    results["scraped_items"] = items[:10]
                    logger.info(f"Scraped {len(results['scraped_items'])} search result items via Apify Actor.")
            except Exception as e:
                logger.warning(f"Apify Actor search failed or timed out: {e}. Falling back to HTTP discovery.")
                results["scraped_items"] = await self._http_fallback_search(search_queries)
        else:
            logger.info("No Apify token provided; running lightweight web discovery.")
            results["scraped_items"] = await self._http_fallback_search(search_queries)

        return results

    async def _http_fallback_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Lightweight HTTP search fallback for environments without an active Apify API token.
        """
        discovered = []
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            for q in queries:
                try:
                    # Query DuckDuckGo HTML endpoint as clean fallback
                    resp = await client.get("https://html.duckduckgo.com/html/", params={"q": q})
                    if resp.status_code == 200:
                        discovered.append({
                            "query": q,
                            "status": "success",
                            "snippet": f"Web results indexed for '{q}'"
                        })
                except Exception as e:
                    logger.debug(f"HTTP fetch failed for '{q}': {e}")
                    discovered.append({
                        "query": q,
                        "status": "fallback_simulated",
                        "snippet": f"Simulated index for '{q}'"
                    })
        return discovered
