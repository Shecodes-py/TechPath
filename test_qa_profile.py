import asyncio
import json
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from src.schemas import LearnerInput
from src.roadmap_builder import TechPathOrchestrator


async def test_qa():
    print("=========================================================")
    print("🧪 Testing TechPath with Quality Tester Profile!")
    print("=========================================================\n")

    qa_input = LearnerInput(
        goal="Quality Tester",
        currentLevel="Beginner",
        knownSkills=["Basic Python"],
        hoursPerWeek=7,
        location="Nigeria",
        interests=["fintech"],
        learningPreferences=["hands-on"],
        depth="deep"
    )

    orchestrator = TechPathOrchestrator(qa_input)
    output = await orchestrator.build_techpath()

    print("---------------------------------------------------------")
    print("SUCCESS: Quality Tester TechPath Dataset Generated!")
    print("---------------------------------------------------------\n")

    print(f"[Target Goal] {output.targetRole}")
    print(f"[Summary Pitch]\n{output.summaryPitch}\n")

    print("[Identified Skill Gaps for Quality Tester]:")
    for gap in output.skillGaps:
        print(f"  * [{gap.category}] {gap.skill} -> Required for: {gap.isRequiredFor}")
        print(f"    Reasoning: {gap.explanation}")
    print()

    print("[Generated Capstone Project Brief]:")
    print(f"  Title: {output.capstone.title}")
    print(f"  Domain Target: {output.capstone.domainTarget}")
    print(f"  Problem: {output.capstone.problemStatement[:140]}...")

    # Save to qa_output.json
    with open("qa_output.json", "w", encoding="utf-8") as f:
        json.dump(output.model_dump(), f, indent=2)

    print("\n=========================================================")
    print("📁 QA output exported to qa_output.json")
    print("=========================================================")


if __name__ == "__main__":
    asyncio.run(test_qa())
