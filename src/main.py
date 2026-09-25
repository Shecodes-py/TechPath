import asyncio
import logging
from apify import Actor

from src.report import build_html_report
from src.schemas import SECRET_INPUT_FIELDS, LearnerInput
from src.roadmap_builder import TechPathOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TechPath.Main")


async def main() -> None:
    """
    Main entry point for TechPath Learning Intelligence Apify Actor.

    Dataset rows are one item per roadmap stage so Console PPE
    (apify-default-dataset-item) can charge per stage if enabled.
    Full JSON is stored as OUTPUT; a readable page is stored as REPORT.
    """
    async with Actor:
        actor_input = await Actor.get_input() or {}
        safe_log = {k: v for k, v in actor_input.items() if k not in SECRET_INPUT_FIELDS}
        logger.info(f"Received Actor input: {safe_log}")

        try:
            learner = LearnerInput(**actor_input)
        except Exception as e:
            logger.error(f"Invalid input provided: {e}")
            await Actor.fail(status_message=f"Invalid input: {e}")
            return

        orchestrator = TechPathOrchestrator(learner)
        output = await orchestrator.build_techpath()
        output_dict = output.model_dump()

        for stage in output.roadmap:
            await Actor.push_data(
                {
                    "recordType": "roadmap_stage",
                    "targetRole": output.targetRole,
                    "generatedAt": output.generatedAt,
                    **stage.model_dump(),
                }
            )

        html_report = build_html_report(output)
        await Actor.set_value("REPORT", html_report, content_type="text/html")

        # Actor-to-Actor Integration: Send report via apify/send-email
        if learner.email:
            apify_token = learner.apifyToken or os.getenv("APIFY_TOKEN")
            if apify_token:
                try:
                    from apify_client import ApifyClient
                    logger.info(f"Triggering Actor-to-Actor call (apify/send-email) for {learner.email}...")
                    client = ApifyClient(token=apify_token)
                    client.actor("apify/send-email").call(
                        run_input={
                            "to": learner.email,
                            "subject": f"Your TechPath Learning Roadmap: {output.targetRole}",
                            "html": html_report,
                        }
                    )
                    logger.info(f"Successfully sent TechPath report to {learner.email} via apify/send-email!")
                except Exception as err:
                    logger.warning(f"Actor-to-Actor email delivery notice: {err}")

        logger.info("TechPath finished. Full JSON in OUTPUT; HTML report in REPORT.")


if __name__ == "__main__":
    asyncio.run(main())
