import asyncio
import json
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from src.schemas import LearnerInput
from src.roadmap_builder import TechPathOrchestrator


async def test_different_profile():
    print("=========================================================")
    print("🧪 Testing TechPath with a COMPLETELY DIFFERENT Profile!")
    print("=========================================================\n")

    # Example Profile 2: Intermediate Backend / Software Engineer in Kenya interested in Fintech
    custom_input = LearnerInput(
        goal="Backend Software Engineer",
        currentLevel="Intermediate",
        knownSkills=["Python", "SQL", "Git"],
        hoursPerWeek=12,
        location="Kenya",
        interests=["Fintech", "Distributed Microservices"],
        learningPreferences=["hands-on", "docs"],
        depth="deep",
    )

    print(f"[Target Goal] {custom_input.goal}")
    print(f"[Current Level] {custom_input.currentLevel} | Location: {custom_input.location}")
    print(f"[Known Skills] {custom_input.knownSkills}")
    print(f"[Niche Interests] {custom_input.interests}\n")

    orchestrator = TechPathOrchestrator(custom_input)
    output = await orchestrator.build_techpath()

    print("---------------------------------------------------------")
    print("SUCCESS: Custom TechPath Dataset Generated!")
    print("---------------------------------------------------------\n")

    print(f"[Summary Pitch]\n{output.summaryPitch}\n")

    print("[Identified Skill Gaps for Backend Engineer]:")
    for gap in output.skillGaps:
        print(f"  * [{gap.category}] {gap.skill} -> Required for: {gap.isRequiredFor}")
    print()

    print("[Generated Capstone Project Brief]:")
    print(f"  Title: {output.capstone.title}")
    print(f"  Domain Target: {output.capstone.domainTarget}")
    print(f"  Problem: {output.capstone.problemStatement[:140]}...")

    # Save to custom_output.json
    with open("custom_output.json", "w", encoding="utf-8") as f:
        json.dump(output.model_dump(), f, indent=2)

    print("\n=========================================================")
    print("📁 Custom output exported to custom_output.json")
    print("=========================================================")


if __name__ == "__main__":
    asyncio.run(test_different_profile())
