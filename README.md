# TechPath Learning Intelligence

## What can TechPath do?

TechPath **builds a personalized, dependency-aware learning roadmap from across the open web** in a single tool: real courses, tutorials, and videos for your goal, staged from your current level to interview-ready, plus quizzes, projects, and matched opportunities. Add a goal with your current level and you can:

| 📦 What you can build | ⚙️ Features & integrations |
|---|---|
| 🗺️ **Staged roadmaps** from beginner foundations to advanced, interview-ready skill | 🔎 **Live discovery** across search results and YouTube, not a static curated list |
| 📝 **Quizzes** per stage to check you're actually ready to move on | 📅 **Freshness-aware** — every resource is discovered at run time, not hardcoded |
| 🛠️ **Projects and a capstone brief** — problem statement, milestones, rubric | ⬇️ **Export data** to JSON, CSV, or Excel |
| 🎯 **Opportunity matches** — hackathons, internships fit to your current stage, not just the most popular one | 🔌 **API access** via the Apify API and CLI |
| 📄 **Human-readable report** as well as structured data | 🤖 **LLM-structured** output, built for MCP, Claude, Gemini, or ChatGPT to consume downstream |

If you're a beginner without a mentor, an intermediate learner who doesn't know what's next, a career switcher, a student prepping for interviews, or an educator building a curriculum, you can get a full personalized path from a single tool.

## What data can I get from TechPath?

TechPath is the structure the open web is missing — the roadmap, the proof-of-readiness, and the next real opportunity behind any tech-learning goal:

- **Roadmap stages:** *what to learn, and in what order.* Foundations → Core Skills → Specialization → Build → Prove → Advance, each with the specific skills that stage covers.
- **Resources:** *real, live links — not a stale list.* Articles, docs, and videos discovered from the web at the moment you run it, each with its title, URL, and type.
- **Quizzes:** *proof you're ready to move on.* A short skill check per stage — conceptual questions, coding prompts, or mini tasks.
- **Projects:** *proof you can build it.* A practical project per stage, plus a full capstone brief — problem statement, objectives, stack, milestones, evaluation rubric — for your target role.
- **Opportunity matches:** *what's actually next for you.* Hackathons, internships, and competitions matched to your current stage, with the reason each one fits.

You can generate a roadmap from just a goal and level, or refine it with your known skills, hours per week, and interests for a more personalized path.

## How It Works — Step by Step

### 1. You submit a learner profile

You fill in your `goal`, `currentLevel`, and — optionally — your `knownSkills`, `hoursPerWeek`, and `interests`. Apify auto-generates this form directly from the input schema, so no coding is needed to try it from the Console.

<img width="979" height="1279" alt="189096" src="https://github.com/user-attachments/assets/8d996c4e-5ec9-4936-b8b2-e2ded015a79b" />


### 2. TechPath calls other Actors to discover real resources

Once you click Start, TechPath's own code (`main.py`) runs on Apify's servers. The first thing it does is call two other Actors already published on the Apify Store, as sub-runs:

- `apify/google-search-scraper` — searches the live web for courses, tutorials, and roadmap content matching your goal
- `streamers/youtube-scraper` — searches YouTube for relevant tutorial videos

This is genuine Actor-to-Actor composition, not a single monolithic scraper: TechPath's code calls `Actor.call(actor_id=..., run_input=...)`, which starts each of those Actors as its own independent run, waits for it to finish, and pulls its dataset back in. Nothing here is hardcoded — every resource returned is whatever those Actors find live, at the exact moment you run TechPath.

![Actor composition screenshot](./screenshots/actor-source-calls.png)
*Screenshot to insert: either the `main.py` source showing the `Actor.call()` lines, or the Console "Runs" view showing the nested `google-search-scraper` and `youtube-scraper` runs kicking off underneath your TechPath run.*

### 3. The LLM structures the raw data into a roadmap

Raw scraped titles, URLs, and descriptions are just a messy list on their own — not a roadmap. TechPath sends that raw data, together with your profile, to an LLM with strict instructions: build a staged roadmap (Foundations → Core Skills → Specialization → Build → Prove → Advance), generate a short quiz and a practical project for every stage, and only ever reference resources that actually appeared in the scraped data — never invent a URL. The LLM returns this as one structured JSON object, which TechPath validates before using it.

### 4. Results are saved two ways

Each roadmap stage is pushed to the Actor's dataset as its own row — one `Actor.push_data()` call per stage — giving you structured, machine-readable output. At the same time, TechPath renders a styled HTML version of the same roadmap and saves it to the key-value store under the key `REPORT`, so a non-technical learner can open one link and read a clean page instead of raw JSON.

![Output schema / dataset screenshot](./screenshots/output-dataset.png)
*Screenshot to insert: the Storage → Dataset tab, showing the actual roadmap-stage rows TechPath returned for a real 

<img width="1080" height="1348" alt="189037" src="https://github.com/user-attachments/assets/1f61e5dc-5523-49f5-9073-f8db34d86d2b" />


### 5. You're charged per what was actually produced

Because every dataset row comes from a real `push_data()` call, turning on the `apify-default-dataset-item` synthetic PPE event means every roadmap stage automatically generates a small charge — no separate billing code anywhere in `main.py`. Turning on `apify-actor-start` charges once per run, the moment it begins.

## Mode 1: Quick roadmap

Enter your **target role** and **current level**, and TechPath builds a full staged roadmap using sensible defaults for hours per week and interests.

- `goal`, e.g. `"Machine Learning Engineer"`
- `currentLevel`: `beginner`, `intermediate`, or `advanced`

## Mode 2: Personalized deep roadmap

Add your **known skills**, **hours per week**, and **interests**, and TechPath tailors resource selection, pacing, and the capstone project to you specifically — e.g. an ML roadmap that leans into healthcare AI because that's what you told it you care about.

| Add this field | And you get |
|---|---|
| `knownSkills` | Skill gaps calculated against your actual starting point, not a generic beginner |
| `hoursPerWeek` | A pace-aware roadmap rather than a one-size-fits-all timeline |
| `interests` | Specialization stages and capstone project themed to what you actually want to build |

## Output and setup

Results land in a **dataset** — one row per roadmap stage. View it as a table in the Storage tab, download as **JSON, CSV, or Excel**, or pull it via the API. A parallel **human-readable HTML report** is saved to the key-value store under `REPORT`, so non-technical learners never have to read raw JSON.

Dataset sample:

```json
{
  "stage": "Core ML",
  "skills": ["supervised learning", "scikit-learn"],
  "resources": [
    {"title": "Intro to Scikit-learn", "url": "https://...", "type": "article"}
  ],
  "quiz": [
    {"question": "What is overfitting?", "answer": "..."}
  ],
  "project": "Build and evaluate a classification model on a public dataset"
}
```

## How much will TechPath cost?

TechPath uses a **pay-per-event (PPE)** pricing model — you're charged for what the run actually produces, not a flat run fee:

| Event | What it means |
|---|---|
| `apify-actor-start` | Charged once when your roadmap run starts |
| `apify-default-dataset-item` | Charged per roadmap stage returned |

*(Exact prices are set in the Monetization tab — update this section once pricing is finalized.)*

## Team

Udotong Peace 

Mayowa Titilola 

Annie

**Track:** EduTech — Access to education and scholarships
**Built for:** Apify x She Code Africa BuildHer Hackathon 2026 — *Ship and Earn Africa*
