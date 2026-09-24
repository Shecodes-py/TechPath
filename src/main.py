import asyncio
import logging
from apify import Actor

from src.schemas import LearnerInput
from src.roadmap_builder import TechPathOrchestrator

# Configure standard logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TechPath.Main")


async def main() -> None:
    """
    Main entry point for TechPath Learning Intelligence Apify Actor.
    """
    async with Actor:
        # Retrieve input payload from Apify Key-Value store
        actor_input = await Actor.get_input() or {}
        logger.info(f"Received Actor input: {actor_input}")

        # Parse and validate input
        try:
            learner = LearnerInput(**actor_input)
        except Exception as e:
            logger.error(f"Invalid input provided: {e}")
            learner = LearnerInput()

        # Orchestrate web discovery and AI roadmap generation
        orchestrator = TechPathOrchestrator(learner)
        output = await orchestrator.build_techpath()

        # Output payload dictionary
        output_dict = output.model_dump()

        # Push structured dataset rows to Apify Dataset storage
        await Actor.push_data(output_dict)

        # Also save primary result object to Apify Key-Value Store under 'OUTPUT'
        await Actor.set_value("OUTPUT", output_dict)

        logger.info("TechPath Learning Intelligence Actor execution finished successfully.")


if __name__ == "__main__":
    asyncio.run(main())
