import asyncio
import json
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from src.schemas import LearnerInput
from src.roadmap_builder import TechPathOrchestrator


async def test_devops():
    print("=========================================================")
    print("☁️ Testing TechPath with Cloud & DevOps Engineer Profile!")
    print("=========================================================\n")

    devops_input = LearnerInput(
        goal="Cloud & DevOps Engineer",
        currentLevel="Beginner",
        knownSkills=["Linux basics", "Bash"],
        hoursPerWeek=10,
        location="Nigeria",
        interests=["Cloud Infrastructure", "Kubernetes"],
        learningPreferences=["hands-on", "project-based"],
        depth="deep"
    )

    orchestrator = TechPathOrchestrator(devops_input)
    output = await orchestrator.build_techpath()

    print("---------------------------------------------------------")
    print("SUCCESS: Cloud & DevOps TechPath Dataset Generated!")
    print("---------------------------------------------------------\n")

    print(f"[Target Goal] {output.targetRole}")
    print(f"[Summary Pitch]\n{output.summaryPitch}\n")

    print("[Identified Skill Gaps for Cloud & DevOps]:")
    for gap in output.skillGaps:
        print(f"  * [{gap.category}] {gap.skill} -> Required for: {gap.isRequiredFor}")
        print(f"    Reasoning: {gap.explanation}")
    print()

    print("[Generated Capstone Project Brief]:")
    print(f"  Title: {output.capstone.title}")
    print(f"  Domain Target: {output.capstone.domainTarget}")
    print(f"  Problem: {output.capstone.problemStatement[:140]}...")

    # Save to devops_output.json
    with open("devops_output.json", "w", encoding="utf-8") as f:
        json.dump(output.model_dump(), f, indent=2)

    print("\n=========================================================")
    print("📁 DevOps output exported to devops_output.json")
    print("=========================================================")


if __name__ == "__main__":
    asyncio.run(test_devops())
