import asyncio
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.schemas import LearnerInput
from src.roadmap_builder import TechPathOrchestrator


async def test_e2e_query(query: str, domain: str):
    print("=========================================================")
    print(f"🚀 END-TO-END TEST: Query = '{query}' (Domain = '{domain}')")
    print("=========================================================\n")

    learner = LearnerInput(
        goal=query,
        currentLevel="Beginner",
        knownSkills=["Python basics"],
        hoursPerWeek=10,
        location="Nigeria",
        interests=[domain],
        learningPreferences=["video", "hands-on"],
        depth="deep"
    )

    orchestrator = TechPathOrchestrator(learner)
    output = await orchestrator.build_techpath()

    print("---------------------------------------------------------")
    print(f"SUCCESS: Generated TechPath Output for '{query}'!")
    print("---------------------------------------------------------\n")

    print(f"[Target Role] {output.targetRole}")
    print(f"[Summary Pitch]\n{output.summaryPitch}\n")

    print(f"[Identified Skill Gaps] ({len(output.skillGaps)} items):")
    for gap in output.skillGaps:
        print(f"  * [{gap.category}] {gap.skill} -> Required for: {gap.isRequiredFor}")
    print()

    print(f"[Roadmap Stages] ({len(output.roadmap)} stages):")
    for stage in output.roadmap[:3]:
        print(f"  * Stage {stage.stageNumber}: {stage.stageName}")
        print(f"    Description: {stage.description}")
        print(f"    Focus Topics: {', '.join(stage.focusTopics)}")
        print(f"    Resources: {len(stage.resources)} items")
    print()

    print(f"[Capstone Brief]:")
    print(f"  Title: {output.capstone.title}")
    print(f"  Domain: {output.capstone.domainTarget}")
    print(f"  Problem: {output.capstone.problemStatement[:120]}...")
    print()

    print(f"[Discovered Sources & External Links] ({len(output.sources)} items):")
    for src in output.sources[:5]:
        print(f"  * {src}")
    print()

    print("=========================================================\n")


async def main():
    await test_e2e_query("Python FastAPI", "REST APIs")
    await test_e2e_query("Machine Learning for beginners", "Computer Vision")


if __name__ == "__main__":
    asyncio.run(main())
