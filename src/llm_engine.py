import os
import json
import logging
from typing import Dict, Any, Optional
from google import genai
from google.genai import types

from src.schemas import LearnerInput

logger = logging.getLogger("TechPath.LLMEngine")


class LLMEngine:
    """
    LLM Intelligence Engine powered by Google Gemini API (Free Tier ready).
    Generates structured learning graph, gap analysis, assessments, projects, 
    and opportunity matches. Includes built-in fallback engine when API keys are absent.
    """

    SYSTEM_PROMPT = """
You are TechPath Learning Intelligence, an expert AI career architect and curriculum design engine.
Your mission: Turn scattered technical learning resources into a structured, dependency-aware development path from beginner to job-ready.

CORE DESIGN PRINCIPLE: Prerequisites before trends.
Never recommend advanced or trending topics (such as LLMs, RAG, AI Agents, or MLOps) unless foundational prerequisites (Python/programming, Data Structures, Linear Algebra/Stats, and basic ML algorithms) are explicitly present in the learner's known skills or assigned to earlier stages.

OUTPUT STRUCTURE REQUIRED (JSON):
Return a valid JSON object with the following top-level keys:
- targetRole (string)
- summaryPitch (string summarizing the path reasoning)
- skillGaps (array of objects: {skill, category, isRequiredFor, confidence, explanation})
- roadmap (array of 6 stage objects:
    stageNumber (1 to 6),
    stageName (e.g., "1. Foundations", "2. Core Skills", "3. Specialization", "4. Build", "5. Prove", "6. Advance"),
    description,
    focusTopics (list),
    prerequisitesRequired (list),
    resources (list of objects: {title, url, resourceType, difficulty, estimatedHours, skillsTaught, prerequisites, summary, isFree}),
    assessments (list of objects: {id, stage, questionType, question, options, correctAnswerOrRubric, explanation}),
    projects (list of objects: {id, title, stage, projectType, problemStatement, objectives, skillsTested, suggestedStack, deliverables, evaluationRubric})
  )
- capstone (object: {title, domainTarget, problemStatement, objectives, skillsTested, suggestedStack, deliverables, milestones, evaluationCriteria, extensionIdeas})
- opportunities (array of objects: {title, organizer, opportunityType, deadline, eligibility, remoteStatus, requiredSkills, difficultyEstimate, applicationUrl, sourceUrl, matchReason})
- sources (array of URLs used or referenced)
"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Initialized Gemini Client successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client: {e}. Will use fallback engine.")

    def generate_intelligence(self, learner: LearnerInput, web_resources: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate structured learning path, assessments, capstone, and opportunities.
        """
        if self.client:
            try:
                return self._call_gemini(learner, web_resources)
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}. Switching to deterministic fallback engine.")
                return self._fallback_generation(learner, web_resources)
        else:
            logger.info("No Gemini API key supplied. Executing deterministic TechPath Intelligence engine.")
            return self._fallback_generation(learner, web_resources)

    def _call_gemini(self, learner: LearnerInput, web_resources: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        user_prompt = f"""
Analyze this learner profile and generate their structured TechPath:

Learner Profile:
- Target Goal/Role: {learner.goal}
- Current Level: {learner.currentLevel}
- Known Skills: {json.dumps(learner.knownSkills)}
- Hours per Week: {learner.hoursPerWeek}
- Location: {learner.location}
- Domain/Niche Interests: {json.dumps(learner.interests)}
- Learning Preferences: {json.dumps(learner.learningPreferences)}
- Roadmap Depth: {learner.depth}

Scraped / Discovered Web Context:
{json.dumps(web_resources) if web_resources else "Standard curated technical web search index available."}

Generate the complete JSON response matching the required structure.
"""
        models_to_try = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-flash", "gemini-pro"]
        last_exception = None

        for model in models_to_try:
            try:
                logger.info(f"Invoking Gemini model: {model}")
                response = self.client.models.generate_content(
                    model=model,
                    contents=self.SYSTEM_PROMPT + "\n\n" + user_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    )
                )
                text = response.text
                return json.loads(text)
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                last_exception = e

        raise last_exception or RuntimeError("All Gemini models failed.")

    def _fallback_generation(self, learner: LearnerInput, web_resources: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        High-quality, rule-driven deterministic generator aligned with PRD specifications.
        """
        goal = learner.goal
        known = set([s.lower() for s in learner.knownSkills])
        interests = learner.interests or ["Healthcare AI"]
        niche = interests[0] if interests else "AI Applications"

        # 1. Skill Gaps
        skill_gaps = []
        if "python" not in str(known) and "programming" not in str(known):
            skill_gaps.append({
                "skill": "Python Programming & Fundamentals",
                "category": "Foundation",
                "isRequiredFor": "Data Structures, ML, and AI Engineering",
                "confidence": 0.95,
                "explanation": "Essential foundation for writing clean, efficient technical code."
            })
        if "math" not in str(known) and "linear algebra" not in str(known):
            skill_gaps.append({
                "skill": "Linear Algebra & Statistics for ML",
                "category": "Foundation",
                "isRequiredFor": "Supervised Learning, Optimization & Model Evaluation",
                "confidence": 0.90,
                "explanation": "Prerequisite for understanding model weights, loss functions, and probability."
            })
        if "dsa" not in str(known) and "data structures" not in str(known):
            skill_gaps.append({
                "skill": "Data Structures & Algorithms",
                "category": "Core",
                "isRequiredFor": "System Efficiency & Coding Interviews",
                "confidence": 0.88,
                "explanation": "Critical for writing scalable algorithms and building production systems."
            })

        # 2. Roadmap Stages
        roadmap = [
            {
                "stageNumber": 1,
                "stageName": "1. Foundations",
                "description": "Master core programming, version control, and prerequisite mathematics.",
                "focusTopics": ["Python Basics", "Git & GitHub", "NumPy & Data Manipulation", "Basic Statistics"],
                "prerequisitesRequired": ["None"],
                "resources": [
                    {
                        "title": "Python for Everybody Specialization",
                        "url": "https://www.py4e.com/",
                        "resourceType": "course",
                        "difficulty": "Beginner",
                        "estimatedHours": 15,
                        "skillsTaught": ["Python syntax", "Data Structures", "Web Scraping"],
                        "prerequisites": [],
                        "summary": "Comprehensive beginner introduction to Python programming.",
                        "isFree": True
                    },
                    {
                        "title": "Git and GitHub for Beginners - Crash Course",
                        "url": "https://www.youtube.com/watch?v=RGOj5yH7evE",
                        "resourceType": "youtube",
                        "difficulty": "Beginner",
                        "estimatedHours": 2,
                        "skillsTaught": ["Git", "Version Control", "GitHub repositories"],
                        "prerequisites": [],
                        "summary": "Hands-on tutorial on tracking code and working with GitHub.",
                        "isFree": True
                    }
                ],
                "assessments": [
                    {
                        "id": "quiz-stage-1",
                        "stage": "1. Foundations",
                        "questionType": "coding_prompt",
                        "question": "Write a Python function `filter_even_squares(numbers)` that takes a list of integers, filters out odd numbers, squares the even numbers, and returns the result in reverse order. Explain the time and space complexity.",
                        "correctAnswerOrRubric": "Function uses list comprehension or loop; time complexity O(N), space complexity O(N).",
                        "explanation": "Tests fundamental Python sequence operations, list manipulation, and algorithmic complexity awareness."
                    }
                ],
                "projects": [
                    {
                        "id": "proj-stage-1",
                        "title": "Automated Data Processing & Extraction CLI",
                        "stage": "1. Foundations",
                        "projectType": "mini_project",
                        "problemStatement": "Build a command-line script that ingests CSV/JSON dataset files, performs validation checks, and outputs formatted statistics.",
                        "objectives": ["Demonstrate proficiency in Python file I/O", "Implement error handling", "Use NumPy/Pandas data structures"],
                        "skillsTested": ["Python", "Pandas", "CLI design"],
                        "suggestedStack": ["Python 3.11", "Pandas", "Argparse"],
                        "deliverables": ["Python CLI script", "Sample dataset", "README documentation"],
                        "evaluationRubric": ["Correct CLI argument handling", "Clean code modularity", "Robust error logging"]
                    }
                ]
            },
            {
                "stageNumber": 2,
                "stageName": "2. Core Skills",
                "description": "Build solid fundamentals in Data Structures, Data Analysis, and Supervised Machine Learning.",
                "focusTopics": ["Data Structures & Algorithms", "SQL & Database Queries", "Supervised Learning", "Scikit-Learn"],
                "prerequisitesRequired": ["Python Programming", "Basic Math"],
                "resources": [
                    {
                        "title": "Scikit-Learn Official Tutorials & User Guide",
                        "url": "https://scikit-learn.org/stable/tutorial/index.html",
                        "resourceType": "documentation",
                        "difficulty": "Intermediate",
                        "estimatedHours": 10,
                        "skillsTaught": ["Scikit-Learn", "Model Training", "Evaluation Metrics"],
                        "prerequisites": ["Python basics", "NumPy"],
                        "summary": "Official step-by-step guide to supervised and unsupervised machine learning algorithms.",
                        "isFree": True
                    }
                ],
                "assessments": [
                    {
                        "id": "quiz-stage-2",
                        "stage": "2. Core Skills",
                        "questionType": "conceptual",
                        "question": "Explain overfitting in machine learning. How do train/test split, cross-validation, and regularization prevent it?",
                        "correctAnswerOrRubric": "Overfitting occurs when model memorizes training noise. Prevention includes cross-validation, L1/L2 regularization, and early stopping.",
                        "explanation": "Verifies core understanding of ML generalization and evaluation."
                    }
                ],
                "projects": [
                    {
                        "id": "proj-stage-2",
                        "title": "Supervised Prediction Engine & Evaluation Pipeline",
                        "stage": "2. Core Skills",
                        "projectType": "integration_project",
                        "problemStatement": "Develop an end-to-end classification pipeline that cleans tabular data, engineers features, trains multiple baseline models, and evaluates ROC-AUC.",
                        "objectives": ["Feature engineering", "Hyperparameter tuning", "Model selection"],
                        "skillsTested": ["Scikit-Learn", "Feature Scaling", "Cross-Validation"],
                        "suggestedStack": ["Python", "Scikit-Learn", "Matplotlib"],
                        "deliverables": ["Jupyter notebook", "Trained model pkl", "Performance report"],
                        "evaluationRubric": ["Proper cross-validation setup", "No data leakage", "Clear metric interpretation"]
                    }
                ]
            },
            {
                "stageNumber": 3,
                "stageName": "3. Specialization",
                "description": "Dive deep into modern Deep Learning, Neural Networks, PyTorch, and domain application in " + niche + ".",
                "focusTopics": ["Neural Networks & Backpropagation", "PyTorch", "Computer Vision / NLP", niche],
                "prerequisitesRequired": ["Core ML Algorithms", "Linear Algebra", "Python"],
                "resources": [
                    {
                        "title": "Deep Learning Specialization - DeepLearning.AI",
                        "url": "https://www.coursera.org/specializations/deep-learning",
                        "resourceType": "course",
                        "difficulty": "Intermediate",
                        "estimatedHours": 30,
                        "skillsTaught": ["PyTorch", "Neural Networks", "Convolutional Networks", "Transformers"],
                        "prerequisites": ["Python", "Linear Algebra"],
                        "summary": "Gold-standard course covering deep learning architectures and optimization.",
                        "isFree": True
                    }
                ],
                "assessments": [
                    {
                        "id": "quiz-stage-3",
                        "stage": "3. Specialization",
                        "questionType": "coding_prompt",
                        "question": "Implement a custom PyTorch `nn.Module` for a 3-layer feedforward network with ReLU activation and Dropout. Write the training loop with CrossEntropyLoss.",
                        "correctAnswerOrRubric": "Module defines __init__ and forward methods; loss calculation and optimizer.step() inside loop.",
                        "explanation": "Validates hands-on PyTorch architecture setup and training dynamics."
                    }
                ],
                "projects": [
                    {
                        "id": "proj-stage-3",
                        "title": niche + " Diagnostic Model Implementation",
                        "stage": "3. Specialization",
                        "projectType": "portfolio_project",
                        "problemStatement": f"Build a PyTorch model specifically tailored for {niche}, processing complex data inputs to predict outcomes.",
                        "objectives": [f"Apply PyTorch to {niche}", "Optimize training loss", "Handle class imbalance"],
                        "skillsTested": ["PyTorch", "Domain Data Handling", "Model Fine-tuning"],
                        "suggestedStack": ["PyTorch", "Torchvision / HuggingFace", "FastAPI"],
                        "deliverables": ["Code repository", "Trained weights", "FastAPI inference service"],
                        "evaluationRubric": ["Functional model training", "Domain validation metrics", "API inference efficiency"]
                    }
                ]
            },
            {
                "stageNumber": 4,
                "stageName": "4. Build",
                "description": "Transform standalone models into robust, deployed production microservices and REST APIs.",
                "focusTopics": ["FastAPI / Flask", "Docker Containerization", "API Deployment", "Model Serving"],
                "prerequisitesRequired": ["PyTorch/Model Training", "Python"],
                "resources": [
                    {
                        "title": "Deploying Machine Learning Models with FastAPI & Docker",
                        "url": "https://fastapi.tiangolo.com/tutorial/",
                        "resourceType": "documentation",
                        "difficulty": "Intermediate",
                        "estimatedHours": 8,
                        "skillsTaught": ["FastAPI", "Async endpoints", "Docker deployment"],
                        "prerequisites": ["Python"],
                        "summary": "Production guide for serving ML model predictions via REST APIs.",
                        "isFree": True
                    }
                ],
                "assessments": [
                    {
                        "id": "quiz-stage-4",
                        "stage": "4. Build",
                        "questionType": "conceptual",
                        "question": "What is the difference between batch inference and real-time API inference? How do Docker containers ensure reproducibility?",
                        "correctAnswerOrRubric": "Real-time serves low-latency single requests; batch processes bulk data offline. Docker encapsulates OS, dependencies, and code environment.",
                        "explanation": "Tests production deployment and software engineering principles."
                    }
                ],
                "projects": [
                    {
                        "id": "proj-stage-4",
                        "title": "Containerized ML Inference Microservice",
                        "stage": "4. Build",
                        "projectType": "integration_project",
                        "problemStatement": "Wrap your trained model inside a Dockerized FastAPI application with request validation and health endpoints.",
                        "objectives": ["Expose REST API endpoints", "Containerize app with Docker", "Test latency and payload throughput"],
                        "skillsTested": ["FastAPI", "Docker", "Pydantic", "Uvicorn"],
                        "suggestedStack": ["FastAPI", "Docker", "Python 3.11"],
                        "deliverables": ["Dockerfile", "App source code", "Deployed API link or container build"],
                        "evaluationRubric": ["Container compiles without error", "Input validation works", "Clean API documentation"]
                    }
                ]
            },
            {
                "stageNumber": 5,
                "stageName": "5. Prove",
                "description": "Validate your skills in real-world environments through hackathons, open-source contributions, and peer reviews.",
                "focusTopics": ["Hackathon Participation", "Open Source PRs", "Technical Blogging", "Peer Code Reviews"],
                "prerequisitesRequired": ["Model Training", "REST APIs", "Git"],
                "resources": [
                    {
                        "title": "Apify & She Code Africa Open-Source Guidelines",
                        "url": "https://apify.com/store",
                        "resourceType": "documentation",
                        "difficulty": "Intermediate",
                        "estimatedHours": 5,
                        "skillsTaught": ["Apify Actor development", "Cloud deployment", "Web intelligence"],
                        "prerequisites": ["Python / JS", "Git"],
                        "summary": "Guide on building reusable cloud tools and submitting to open source ecosystems.",
                        "isFree": True
                    }
                ],
                "assessments": [
                    {
                        "id": "quiz-stage-5",
                        "stage": "5. Prove",
                        "questionType": "coding_prompt",
                        "question": "Draft an Open Source Pull Request summary detailing a bug fix or feature addition, including reproduction steps and test results.",
                        "correctAnswerOrRubric": "Clear title, problem description, solution implementation details, and verification commands.",
                        "explanation": "Evaluates professional open-source communication and collaboration."
                    }
                ],
                "projects": [
                    {
                        "id": "proj-stage-5",
                        "title": "Apify Open Source Cloud Tool / Actor",
                        "stage": "5. Prove",
                        "projectType": "portfolio_project",
                        "problemStatement": "Publish a reusable Apify Actor that automates web data extraction and provides structured API output.",
                        "objectives": ["Create input schema", "Store dataset output", "Publish to Apify Store"],
                        "skillsTested": ["Apify SDK", "Python", "API design"],
                        "suggestedStack": ["Apify Python SDK", "Docker", "Pydantic"],
                        "deliverables": ["Published Apify Actor", "Public GitHub repository"],
                        "evaluationRubric": ["Valid input/output schema", "Error-free execution", "Comprehensive README"]
                    }
                ]
            },
            {
                "stageNumber": 6,
                "stageName": "6. Advance",
                "description": "Expand into cutting-edge architectures: Large Language Models, RAG Systems, AI Agents, and MLOps.",
                "focusTopics": ["RAG Systems & Vector DBs", "AI Agents & LangChain/LlamaIndex", "MLOps & Model Monitoring"],
                "prerequisitesRequired": ["All Stages 1 to 5 satisfied"],
                "resources": [
                    {
                        "title": "Full Stack LLM & RAG Application Architecture",
                        "url": "https://www.deeplearning.ai/short-courses/",
                        "resourceType": "course",
                        "difficulty": "Advanced",
                        "estimatedHours": 12,
                        "skillsTaught": ["RAG", "Vector Search", "LangChain", "Evaluation"],
                        "prerequisites": ["Python", "PyTorch / Transformers"],
                        "summary": "Advanced building blocks for AI agents, retrieval systems, and LLM applications.",
                        "isFree": True
                    }
                ],
                "assessments": [
                    {
                        "id": "quiz-stage-6",
                        "stage": "6. Advance",
                        "questionType": "conceptual",
                        "question": "How does Retrieval-Augmented Generation (RAG) overcome LLM context window limits and hallucinations? Compare sparse vs dense vector retrieval.",
                        "correctAnswerOrRubric": "RAG fetches relevant chunked documents via embedding cosine similarity to ground response. Dense embeddings capture semantic intent; sparse keywords capture exact terms.",
                        "explanation": "Assesses mastery of modern LLM architecture choices."
                    }
                ],
                "projects": [
                    {
                        "id": "proj-stage-6",
                        "title": "Autonomous AI Agent System with Tool Calling",
                        "stage": "6. Advance",
                        "projectType": "portfolio_project",
                        "problemStatement": "Develop a multi-tool AI Agent capable of web searching, data processing, and generating automated reports.",
                        "objectives": ["Implement tool selection logic", "Enforce safety boundaries", "Manage long-running tasks"],
                        "skillsTested": ["LLM APIs", "Vector Database", "Agentic Workflows"],
                        "suggestedStack": ["Python", "Gemini API", "ChromaDB / Qdrant"],
                        "deliverables": ["Agent architecture codebase", "Live interactive demo"],
                        "evaluationRubric": ["Reliable tool execution", "Minimal hallucination", "Scalable vector index"]
                    }
                ]
            }
        ]

        # 3. Capstone Brief
        capstone = {
            "title": f"Production-Grade {niche} Intelligence & Decision System",
            "domainTarget": f"{goal} - {niche}",
            "problemStatement": f"Develop an end-to-end, deployed AI platform designed for {niche}. The system ingests raw domain data, applies specialized PyTorch models, and exposes a real-time containerized API.",
            "objectives": [
                f"Curate and preprocess realistic datasets for {niche}",
                "Train and evaluate a high-performing PyTorch neural network",
                "Deploy a containerized FastAPI microservice with automated testing",
                "Integrate continuous model monitoring and automated evaluation"
            ],
            "skillsTested": ["Python", "PyTorch", "FastAPI", "Docker", "Scikit-Learn", "Data Pipeline"],
            "suggestedStack": ["Python 3.11", "PyTorch", "FastAPI", "Docker", "Streamlit", "Apify"],
            "deliverables": [
                "Full Git Repository with clean commit history",
                "Dockerfile & docker-compose.yml setup",
                "Live deployed demo (Render/HuggingFace Spaces/AWS)",
                "System architecture document and benchmark report"
            ],
            "milestones": [
                "Milestone 1: Data acquisition and exploratory data analysis",
                "Milestone 2: Baseline ML model vs PyTorch deep learning model",
                "Milestone 3: FastAPI microservice wrapper and Docker containerization",
                "Milestone 4: Interactive frontend dashboard and deployment"
            ],
            "evaluationCriteria": [
                "Model F1-Score / Accuracy on test set (>0.85 target)",
                "API endpoint p95 latency under 200ms",
                "Code coverage and clean engineering standards",
                "Clarity of README and user documentation"
            ],
            "extensionIdeas": [
                "Add RAG pipeline to allow querying domain medical/tech guidelines",
                "Integrate real-time notification alerts for anomalous model outputs"
            ]
        }

        # 4. Opportunities
        opportunities = [
            {
                "title": "Apify × She Code Africa Hackathon 2026",
                "organizer": "Apify & She Code Africa",
                "opportunityType": "hackathon",
                "deadline": "2026-10-15",
                "eligibility": "Open to developers, students, and tech learners across Africa",
                "remoteStatus": "Remote",
                "requiredSkills": ["Python", "Web Crawling", "API Development", "AI/ML"],
                "difficultyEstimate": "Beginner to Intermediate friendly",
                "applicationUrl": "https://apify.com/hackathons",
                "sourceUrl": "https://apify.com/",
                "matchReason": f"Directly aligns with your growth stage and goal ({goal}). Offers mentorship, cash prizes, and Apify Actor publishing exposure."
            },
            {
                "title": "Kaggle Community Healthcare AI Challenge",
                "organizer": "Kaggle Competitions",
                "opportunityType": "competition",
                "deadline": "2026-11-01",
                "eligibility": "Global community",
                "remoteStatus": "Remote",
                "requiredSkills": ["Python", "Scikit-Learn", "PyTorch", "Data Science"],
                "difficultyEstimate": "Intermediate",
                "applicationUrl": "https://www.kaggle.com/competitions",
                "sourceUrl": "https://www.kaggle.com/",
                "matchReason": f"Perfect match for building practical proof in {niche} using real tabular and image datasets."
            },
            {
                "title": "Open Source AI Engineering Fellowship",
                "organizer": "Global Tech Talent Network",
                "opportunityType": "internship",
                "deadline": "2026-12-01",
                "eligibility": "Early-career developers & students",
                "remoteStatus": "Remote",
                "requiredSkills": ["Python", "Git", "REST APIs"],
                "difficultyEstimate": "Intermediate",
                "applicationUrl": "https://shecodeafrica.org/",
                "sourceUrl": "https://shecodeafrica.org/",
                "matchReason": "Provides mentorship bridge into full-time remote engineering roles."
            }
        ]

        return {
            "targetRole": goal,
            "summaryPitch": f"Tailored {goal} roadmap for a {learner.currentLevel} in {learner.location} focusing on {niche}. Designed with strict prerequisite ordering (Foundations → Core → Specialization → Build → Prove → Advance).",
            "skillGaps": skill_gaps,
            "roadmap": roadmap,
            "capstone": capstone,
            "opportunities": opportunities,
            "sources": [
                "https://www.py4e.com/",
                "https://scikit-learn.org/",
                "https://coursera.org/",
                "https://fastapi.tiangolo.com/",
                "https://apify.com/store"
            ]
        }
