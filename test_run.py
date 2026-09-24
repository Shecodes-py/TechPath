import asyncio
import json
import os
import sys
from dotenv import load_dotenv

# Reconfigure stdout for UTF-8 on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load local environment variables from .env if present
load_dotenv()

from src.schemas import LearnerInput
from src.roadmap_builder import TechPathOrchestrator


async def run_demo():
    print("=========================================================")
    print("TechPath Learning Intelligence Actor - Local Test Run")
    print("=========================================================\n")

    # Sample input matching PRD Section 27 Demo Prompt
    demo_input = LearnerInput(
        goal="Machine Learning Engineer",
        currentLevel="Beginner",
        knownSkills=["Python basics"],
        hoursPerWeek=8,
        location="Nigeria",
        interests=["Healthcare AI", "NLP"],
        learningPreferences=["video", "hands-on"],
        depth="deep",
        geminiApiKey=os.getenv("GEMINI_API_KEY"),
        apifyToken=os.getenv("APIFY_TOKEN")
    )

    print(f"[Input Goal] {demo_input.goal}")
    print(f"[Current Level] {demo_input.currentLevel} | Location: {demo_input.location}")
    print(f"[Known Skills] {demo_input.knownSkills}")
    print(f"[Interests] {demo_input.interests}\n")

    orchestrator = TechPathOrchestrator(demo_input)
    output = await orchestrator.build_techpath()

    print("---------------------------------------------------------")
    print("SUCCESS: TechPath Intelligence Dataset Generated!")
    print("---------------------------------------------------------\n")

    print(f"[Summary Pitch]\n{output.summaryPitch}\n")

    print("[Identified Skill Gaps]")
    for gap in output.skillGaps:
        print(f"  * [{gap.category}] {gap.skill} -> Required for: {gap.isRequiredFor}")
    print()

    print("[6-Stage Dependency-Aware Roadmap]")
    for stage in output.roadmap:
        print(f"\n  Stage {stage.stageNumber}: {stage.stageName}")
        print(f"  Description: {stage.description}")
        print(f"  Focus Topics: {', '.join(stage.focusTopics)}")
        print(f"  Resources Count: {len(stage.resources)} | Assessments: {len(stage.assessments)} | Projects: {len(stage.projects)}")

    print(f"\n[Portfolio Capstone Brief]")
    print(f"  Title: {output.capstone.title}")
    print(f"  Domain Target: {output.capstone.domainTarget}")
    print(f"  Problem: {output.capstone.problemStatement[:120]}...")

    print(f"\n[Opportunity Matches] ({len(output.opportunities)} found):")
    for opp in output.opportunities:
        print(f"  * {opp.title} ({opp.opportunityType}) | Deadline: {opp.deadline} | Match: {opp.matchReason[:90]}...")

    # Save summary report
    with open("demo_output.json", "w", encoding="utf-8") as f:
        json.dump(output.model_dump(), f, indent=2)

    print("\n=========================================================")
    print("📁 Full dataset exported to demo_output.json")
    print("=========================================================")


if __name__ == "__main__":
    asyncio.run(run_demo())
