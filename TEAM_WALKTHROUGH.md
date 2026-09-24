# 📘 TechPath Learning Intelligence — Teammate Onboarding & Technical Walkthrough

Welcome to the **TechPath Learning Intelligence** codebase! This document is a complete walkthrough designed to get you up to speed on the project architecture, design choices, codebase structure, and how to run, test, and build on top of what we've created for the **Apify × She Code Africa Hackathon**.

---

## 1. Executive Summary & Vision

### What is TechPath?
**TechPath Learning Intelligence** is a reusable **Apify Actor** (a serverless backend cloud automation program) that converts the scattered technical-learning web into a structured, personalized development pathway.

> 💡 **Core Pitch**: *"The internet has the content. TechPath builds the structure."*

### Why an Apify Actor instead of just a course website?
Most hackathon projects build static course aggregator websites. We built **reusable infrastructure**. TechPath accepts structured JSON inputs (learner goal, current skills, location, time commitment, interests) and returns structured JSON outputs (skill gap analysis, 6-stage roadmap, stage assessments, capstone project brief, and matched real-world hackathons/opportunities). 

Other developers, AI coaches, or web dashboards can invoke our Actor programmatically through the **Apify API**.

---

## 2. Core Architectural Principles

1. **Prerequisites Before Trends**:
   * Learner interested in LLMs, AI Agents, or RAG? TechPath checks whether they know Python programming, Linear Algebra, and Data Structures first.
   * Higher-level topics are labeled as later-stage material until foundational prerequisites are satisfied.

2. **6-Stage Progression Logic**:
   * **Stage 1: Foundations** — Core programming (Python), Git, NumPy, Basic Statistics
   * **Stage 2: Core Skills** — Data Structures & Algorithms, SQL, Supervised Machine Learning
   * **Stage 3: Specialization** — Deep Learning, PyTorch, Domain Applications (*e.g., Healthcare AI*)
   * **Stage 4: Build** — Production microservices, FastAPI REST APIs, Docker containerization
   * **Stage 5: Prove** — Hackathon entry, Open-Source PRs, Apify Actor publishing
   * **Stage 6: Advance** — Vector Databases, RAG pipelines, Autonomous AI Agents, MLOps

3. **Proof of Skill (Assessments & Projects)**:
   * Every stage has conceptual/coding quizzes and stage-reinforcing mini projects.
   * Stage 3/6 produces a portfolio-level **Capstone Project Brief** with problem statements, deliverables, and rubrics.

4. **Opportunity Matching**:
   * Connects learners to real hackathons (e.g. *Apify × She Code Africa Hackathon 2026*), Kaggle challenges, and open-source fellowships tailored to their growth stage and location.

---

## 3. System Architecture & Data Flow

```
+-------------------------------------------------------------------+
|                        1. LEARNER INPUT                           |
| { goal, currentLevel, knownSkills, location, interests, depth }  |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|                      2. WEB DISCOVERY LAYER                       |
|   (Calls Apify Google Search Scraper or HTTP Web Indexing)        |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|                      3. LLM REASONING LAYER                       |
|   (Google Gemini 3.6/2.5 Flash + Deterministic Fallback Engine)   |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|                   4. STRUCTURED DATASET OUTPUT                    |
| (Saved to Apify Dataset & Key-Value Store for API consumption)   |
+-------------------------------------------------------------------+
```

---

## 4. Codebase Directory Map

Here is how the project files are organized:

```
TechPath/
├── .actor/
│   ├── actor.json           # Actor spec (slug: techpath-learning-intelligence, ver: 0.0)
│   └── input_schema.json    # Apify Console UI input form specification
├── src/
│   ├── __init__.py
│   ├── schemas.py           # Pydantic data models for Inputs & Outputs
│   ├── llm_engine.py        # Gemini API integration & deterministic fallback generator
│   ├── discovery.py         # Google Search Scraper & HTTP web fetcher
│   ├── roadmap_builder.py   # Main orchestrator (combines discovery, LLM, & validation)
│   └── main.py              # Apify Actor entrypoint (handles Actor.push_data & storage)
├── .env.example             # Local environment variables template
├── Dockerfile               # Production container definition (apify/actor-python:3.11)
├── requirements.txt         # Dependencies (apify, google-genai, pydantic, httpx)
├── test_run.py              # Local CLI runner simulating PRD hackathon demo prompt
└── TEAM_WALKTHROUGH.md      # This document!
```

---

## 5. File-by-File Technical Breakdown

### `src/schemas.py`
Contains all **Pydantic v2** models defining the exact data contracts:
* `LearnerInput`: Input payload parameters.
* `SkillGap`: Identified missing prerequisite with confidence score and reasoning.
* `ResourceItem`: Curated course/video/doc item with difficulty, skills taught, and prerequisites.
* `AssessmentQuestion`: Quiz/coding prompt with rubrics.
* `ProjectBrief` & `CapstoneBrief`: Detailed project specs.
* `OpportunityMatch`: Hackathon/internship record with match rationale.
* `TechPathOutput`: Final consolidated output structure.

### `src/llm_engine.py`
Houses the AI reasoning layer:
* **Gemini Client**: Connects via official `google-genai` SDK using `gemini-3.6-flash`, `gemini-2.5-flash`, etc.
* **Deterministic Fallback Engine**: If no Gemini API key is supplied or API rate limits occur, it seamlessly falls back to a rule-driven generator that produces complete, high-quality roadmap output so the Actor never fails.

### `src/discovery.py`
Handles web discovery:
* Attempts to invoke Apify's `apify/google-search-scraper` via `self.client.actor(...).call()`.
* Extracts search results for courses, prerequisites, and hackathons.
* Safely extracts dataset IDs (`getattr(run, "default_dataset_id", None)`).
* Includes DuckDuckGo HTTP fallback if running offline without an Apify API token.

### `src/roadmap_builder.py`
The central orchestrator (`TechPathOrchestrator`):
* Calls `discovery.discover()`.
* Invokes `llm_engine.generate_intelligence()`.
* Validates and parses the raw result into the strongly-typed `TechPathOutput` Pydantic model.

### `src/main.py`
The Apify execution entry point:
* Uses `async with Actor:`.
* Fetches user input via `await Actor.get_input()`.
* Runs the orchestrator.
* Pushes dataset items via `await Actor.push_data(...)` and stores key-value output under `'OUTPUT'`.

---

## 6. How to Run & Test the Project Locally

### 1. Set Up Environment
```powershell
# 1. Clone repository & navigate to folder
cd TechPath

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 2. Run Local CLI Simulation
Run our demo script to see the output report formatted in your terminal:
```powershell
python test_run.py
```
This exports a `demo_output.json` file in the root folder containing the complete generated dataset.

### 3. Deploy Updates to Apify Cloud
To push new code or prompt changes live to Apify:
```powershell
# Ensure Apify CLI is installed (npm install -g apify-cli)
apify login
apify push
```

Live Actor URL: [https://console.apify.com/actors/3cGQUsrX8W99r2eMD](https://console.apify.com/actors/3cGQUsrX8W99r2eMD)

---

## 7. Hackathon Demo & Pitch Strategy for Judges

When presenting TechPath to hackathon judges:

1. **The Problem**: *"Technical learning is scattered. Learners watch tutorials without knowing what prerequisites they're missing or how to prove their skills."*
2. **The Solution**: *"TechPath is an Apify Actor that builds dependency-aware learning paths with skill checks, capstone projects, and hackathon matches."*
3. **The Live Demo**: Show the Actor running live on Apify Console or run `python test_run.py` to display the generated JSON output.
4. **Key Differentiator**: *"We built reusable API infrastructure on Apify, not just a static webpage."*

---

## 8. Ideas for Further Improvements

If you'd like to build further on this codebase, here are great areas to contribute:
* **Add YouTube Transcript Scraper**: Integrate `apify/youtube-scraper` in `src/discovery.py` to extract video timestamps for key topics.
* **Frontend Integration**: Build a simple React/Next.js UI or Streamlit dashboard that calls the Apify API (`https://api.apify.com/v2/acts/3cGQUsrX8W99r2eMD/runs`) and renders the roadmap graphically.
* **Expanded Track Profiles**: Customize template prompts for Cloud/DevOps, Cybersecurity, or Data Engineering tracks.

---
*Happy coding team! Let's win this hackathon! 🚀*
