import logging
from datetime import datetime
from typing import Dict, Any

from src.schemas import LearnerInput, TechPathOutput
from src.llm_engine import LLMEngine
from src.discovery import ResourceDiscovery

logger = logging.getLogger("TechPath.RoadmapBuilder")


class TechPathOrchestrator:
    """
    Main orchestrator for TechPath Learning Intelligence Actor.
    Executes web discovery -> skill graph analysis -> LLM reasoning -> dataset output formatting.
    """

    def __init__(self, learner_input: LearnerInput):
        self.input = learner_input
        self.llm = LLMEngine(api_key=learner_input.geminiApiKey)
        self.discovery = ResourceDiscovery(apify_token=learner_input.apifyToken)

    async def build_techpath(self) -> TechPathOutput:
        """
        Build the full structured TechPath dataset.
        """
        logger.info(f"Starting TechPath generation for Goal: '{self.input.goal}', Level: '{self.input.currentLevel}'")

        # Step 1: Web discovery
        web_context = await self.discovery.discover(
            goal=self.input.goal,
            location=self.input.location,
            interests=self.input.interests
        )

        # Step 2: LLM analysis & roadmap generation
        raw_output = self.llm.generate_intelligence(self.input, web_context)

        # Step 3: Parse and validate into Pydantic schema
        output = TechPathOutput(
            profile=self.input,
            targetRole=raw_output.get("targetRole", self.input.goal),
            summaryPitch=raw_output.get("summaryPitch", f"Personalized pathway to become a {self.input.goal}"),
            skillGaps=raw_output.get("skillGaps", []),
            roadmap=raw_output.get("roadmap", []),
            capstone=raw_output.get("capstone", {}),
            opportunities=raw_output.get("opportunities", []),
            sources=raw_output.get("sources", []),
            generatedAt=datetime.utcnow().isoformat() + "Z"
        )

        logger.info(f"Successfully constructed TechPath with {len(output.roadmap)} stages, {len(output.skillGaps)} skill gaps, and {len(output.opportunities)} opportunity matches.")
        return output
