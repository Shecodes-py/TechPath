import logging
from datetime import datetime, timezone
from typing import Any, Dict

from src.discovery import ResourceDiscovery
from src.llm_engine import LLMEngine
from src.schemas import LearnerInput, TechPathOutput

logger = logging.getLogger("TechPath.RoadmapBuilder")


class TechPathOrchestrator:
    """
    Main orchestrator for TechPath Learning Intelligence Actor.
    Executes web discovery -> skill graph analysis -> LLM reasoning -> dataset output formatting.
    """

    def __init__(self, learner_input: LearnerInput):
        self.input = learner_input
        gemini_key = learner_input.geminiApiKey
        anthropic_key = learner_input.anthropicApiKey
        generic = learner_input.llmApiKey
        provider = learner_input.llmProvider or "auto"
        if generic:
            if provider in ("claude", "anthropic"):
                anthropic_key = anthropic_key or generic
            else:
                gemini_key = gemini_key or generic
        self.llm = LLMEngine(
            api_key=gemini_key,
            anthropic_key=anthropic_key,
            provider=provider,
        )
        self.discovery = ResourceDiscovery(apify_token=learner_input.apifyToken)

    async def build_techpath(self) -> TechPathOutput:
        logger.info(
            f"Starting TechPath generation for Goal: '{self.input.goal}', "
            f"Level: '{self.input.currentLevel}'"
        )

        web_context = await self.discovery.discover(
            goal=self.input.goal,
            location=self.input.location,
            interests=self.input.interests,
            current_level=self.input.currentLevel,
        )

        raw_output = await self.llm.generate_intelligence(self.input, web_context)
        output = self._parse_output(raw_output, web_context)
        if output is None:
            logger.warning("LLM JSON did not match schema; using fallback engine.")
            raw_output = self.llm._fallback_generation(self.input, web_context)
            output = self._parse_output(raw_output, web_context)
            if output is None:
                raise ValueError("Failed to construct a valid TechPath output.")

        logger.info(
            f"Successfully constructed TechPath with {len(output.roadmap)} stages, "
            f"{len(output.skillGaps)} skill gaps, and {len(output.opportunities)} opportunity matches."
        )
        return output

    def _parse_output(
        self, raw_output: Dict[str, Any], web_context: Dict[str, Any]
    ) -> TechPathOutput | None:
        try:
            sources = raw_output.get("sources") or []
            vids = web_context.get("video_results", [])
            search_hits = web_context.get("search_results", [])
            return TechPathOutput(
                profile=self.input.public_profile(),
                targetRole=raw_output.get("targetRole", self.input.goal),
                summaryPitch=raw_output.get(
                    "summaryPitch", f"Personalized pathway to become a {self.input.goal}"
                ),
                skillGaps=raw_output.get("skillGaps", []),
                roadmap=raw_output.get("roadmap", []),
                capstone=raw_output.get("capstone", {}),
                opportunities=raw_output.get("opportunities", []),
                sources=sources or _source_urls(raw_output, web_context),
                discoveredVideos=vids,
                discoveredSearchResults=search_hits,
                generatedAt=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )
        except Exception as e:
            logger.warning(f"TechPath schema validation failed: {e}")
            return None


def _source_urls(raw_output: Dict[str, Any], web_context: Dict[str, Any]) -> list[str]:
    urls = list(raw_output.get("sources") or [])
    for row in web_context.get("search_results", []) + web_context.get("video_results", []):
        url = row.get("url")
        if url and url not in urls:
            urls.append(url)
    return urls
