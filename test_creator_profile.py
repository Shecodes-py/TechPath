import asyncio
import json
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from src.schemas import LearnerInput
from src.roadmap_builder import TechPathOrchestrator


async def test_creator():
    print("=========================================================")
    print("🎬 Testing TechPath with Content Creator Profile!")
    print("=========================================================\n")

    creator_input = LearnerInput(
        goal="Content creator",
        currentLevel="Beginner",
        knownSkills=["Python basics"],
        hoursPerWeek=8,
        location="Nigeria",
        interests=["videography"],
        learningPreferences=["video", "hands-on"],
        depth="deep"
    )

    orchestrator = TechPathOrchestrator(creator_input)
    output = await orchestrator.build_techpath()

    print("---------------------------------------------------------")
    print("SUCCESS: Content Creator TechPath Dataset Generated!")
    print("---------------------------------------------------------\n")

    print(f"[Target Goal] {output.targetRole}")
    print(f"[Summary Pitch]\n{output.summaryPitch}\n")

    print("[Identified Skill Gaps for Content Creator]:")
    for gap in output.skillGaps:
        print(f"  * [{gap.category}] {gap.skill} -> Required for: {gap.isRequiredFor}")
    print()

    print("[Generated Capstone Project Brief]:")
    print(f"  Title: {output.capstone.title}")
    print(f"  Domain Target: {output.capstone.domainTarget}")
    print(f"  Problem: {output.capstone.problemStatement[:140]}...")

    # Save to creator_output.json
    with open("creator_output.json", "w", encoding="utf-8") as f:
        json.dump(output.model_dump(), f, indent=2)

    print("\n=========================================================")
    print("📁 Creator output exported to creator_output.json")
    print("=========================================================")


if __name__ == "__main__":
    asyncio.run(test_creator())
