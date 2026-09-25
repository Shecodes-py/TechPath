"""
TechPath Backend API Server.

Provides REST endpoints for:
- POST /api/generate        -> Full TechPath Orchestration (Scrape + LLM + Dataset)
- GET /api/search/youtube   -> Independent YouTube Video Retrieval Route
- GET /api/search/web       -> Independent Google/Web Search Retrieval Route
- GET /api/health           -> Backend Status & Environment Variable Audit
"""

import asyncio
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

load_dotenv()

from src.discovery import ResourceDiscovery
from src.roadmap_builder import TechPathOrchestrator
from src.schemas import LearnerInput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TechPath.Server")


class TechPathHTTPHandler(BaseHTTPRequestHandler):

    def _set_cors_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Type", content_type)
        self.end_headers()

    def do_OPTIONS(self):
        self._set_cors_headers(200)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/api/health":
            env_status = {
                "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
                "APIFY_TOKEN": bool(os.getenv("APIFY_TOKEN")),
                "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
                "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            }
            self._set_cors_headers(200)
            self.wfile.write(
                json.dumps(
                    {
                        "status": "healthy",
                        "service": "TechPath Learning Intelligence Backend API",
                        "environmentVariables": env_status,
                    }
                ).encode("utf-8")
            )
            return

        if path == "/api/search/youtube":
            query = params.get("q", ["Python FastAPI"])[0]
            logger.info(f"[Server Route] GET /api/search/youtube?q={query}")
            discovery = ResourceDiscovery()
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                res = loop.run_until_complete(discovery.discover(goal=query, location="Global", interests=[]))
                loop.close()
                self._set_cors_headers(200)
                self.wfile.write(
                    json.dumps(
                        {
                            "status": "success",
                            "query": query,
                            "count": len(res["video_results"]),
                            "video_results": res["video_results"],
                            "integrationStatus": res["integration_status"]["youtube"],
                        }
                    ).encode("utf-8")
                )
            except Exception as e:
                logger.error(f"[Server Route] YouTube search failed: {e}")
                self._set_cors_headers(500)
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return

        if path == "/api/search/web":
            query = params.get("q", ["Python FastAPI documentation"])[0]
            logger.info(f"[Server Route] GET /api/search/web?q={query}")
            discovery = ResourceDiscovery()
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                res = loop.run_until_complete(discovery.discover(goal=query, location="Global", interests=[]))
                loop.close()
                self._set_cors_headers(200)
                self.wfile.write(
                    json.dumps(
                        {
                            "status": "success",
                            "query": query,
                            "count": len(res["search_results"]),
                            "search_results": res["search_results"],
                            "integrationStatus": res["integration_status"]["google_search"],
                        }
                    ).encode("utf-8")
                )
            except Exception as e:
                logger.error(f"[Server Route] Web search failed: {e}")
                self._set_cors_headers(500)
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return

        # Serve static frontend index.html if requested
        if path == "/" or path == "/index.html":
            try:
                with open("index.html", "rb") as f:
                    content = f.read()
                self._set_cors_headers(200, "text/html")
                self.wfile.write(content)
                return
            except Exception as e:
                logger.error(f"Failed to serve index.html: {e}")

        self._set_cors_headers(404)
        self.wfile.write(json.dumps({"status": "error", "message": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            except Exception:
                payload = {}

            logger.info(f"[Server Route] POST /api/generate - Payload: {payload}")
            try:
                learner = LearnerInput(**payload)
                orchestrator = TechPathOrchestrator(learner)

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                output = loop.run_until_complete(orchestrator.build_techpath())
                loop.close()

                out_dict = output.model_dump()
                self._set_cors_headers(200)
                self.wfile.write(json.dumps(out_dict).encode("utf-8"))
            except Exception as e:
                logger.error(f"[Server Route] POST /api/generate failed: {e}")
                self._set_cors_headers(500)
                self.wfile.write(
                    json.dumps(
                        {
                            "status": "error",
                            "message": f"TechPath Generation Error: {str(e)}",
                        }
                    ).encode("utf-8")
                )
            return

        self._set_cors_headers(404)
        self.wfile.write(json.dumps({"status": "error", "message": "Endpoint not found"}).encode("utf-8"))


def run_server(port: int = 8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, TechPathHTTPHandler)
    logger.info(f"=========================================================")
    logger.info(f"🚀 TechPath Backend Server running on http://localhost:{port}")
    logger.info(f"   - POST /api/generate        (Full Orchestration API)")
    logger.info(f"   - GET  /api/search/youtube   (YouTube Video Search API)")
    logger.info(f"   - GET  /api/search/web       (Google/Web Search API)")
    logger.info(f"   - GET  /api/health           (Health & Env Variables Check)")
    logger.info(f"=========================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped.")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    run_server(port)
