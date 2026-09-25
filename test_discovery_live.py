import asyncio
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.discovery import ResourceDiscovery


async def test():
    print("=========================================================")
    print("🔍 Testing Live External Data Retrieval (YouTube & Web)")
    print("=========================================================\n")

    discovery = ResourceDiscovery()
    res = await discovery.discover(
        goal="FastAPI Developer",
        location="Nigeria",
        interests=["Python", "Web APIs"],
        current_level="Beginner"
    )

    print("---------------------------------------------------------")
    print(f"Discovery Mode: {res['discovery_mode']}")
    print(f"Integration Status: {json.dumps(res['integration_status'], indent=2)}")
    print("---------------------------------------------------------\n")

    print(f"▶️ Retrieved {len(res['video_results'])} Real YouTube Videos:")
    for vid in res['video_results'][:3]:
        print(f"  * Title: {vid['title']}")
        print(f"    URL: {vid['url']}")
        print(f"    Video ID: {vid['videoId']}")
        print(f"    Thumbnail: {vid['thumbnailUrl']}")
        print(f"    Channel: {vid['channelName']}")
        print(f"    Published: {vid['publishedAt']}")
        print(f"    Description: {vid['description'][:80]}...")
        print()

    print(f"🌐 Retrieved {len(res['search_results'])} Real Web Search Items:")
    for item in res['search_results'][:3]:
        print(f"  * Title: {item['title']}")
        print(f"    URL: {item['url']}")
        print(f"    Domain: {item['source']}")
        print(f"    Snippet: {item['description'][:100]}...")
        print()


if __name__ == "__main__":
    asyncio.run(test())
