import os
import json
import logging
from typing import Dict, Any, Optional

from src.llm_clients import call_claude, call_gemini_latest, parse_llm_json
from src.schemas import LearnerInput

logger = logging.getLogger("TechPath.LLMEngine")

_PLACEHOLDER_KEYS = {
    "",
    "your_gemini_api_key_here",
    "your_anthropic_api_key_here",
}


def _usable_key(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    trimmed = key.strip()
    if trimmed in _PLACEHOLDER_KEYS or trimmed.startswith("your_"):
        return None
    return trimmed


def _trim_web_context(web_resources: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not web_resources:
        return {"search_results": [], "video_results": []}
    return {
        "search_results": web_resources.get("search_results", [])[:15],
        "video_results": web_resources.get("video_results", [])[:10],
    }


class LLMEngine:
    """
    LLM Intelligence Engine powered by Google Gemini API (Free Tier ready).
    Generates structured learning graph, gap analysis, assessments, projects, 
    and opportunity matches. Includes built-in multi-domain fallback engine.
    """

    SYSTEM_PROMPT = """
You are TechPath Learning Intelligence, an expert AI career architect and curriculum design engine.
Your mission: Turn scattered technical learning resources into a structured, dependency-aware development path from beginner to job-ready.

CORE DESIGN PRINCIPLE: Prerequisites before trends.
Never recommend advanced topics unless foundational prerequisites (core skills, tools, and theory) are explicitly present in the learner's known skills or assigned to earlier stages.

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

RESOURCE RULES:
- Prefer REAL scraped search results and YouTube videos supplied in the user prompt.
- When those lists are non-empty, do not invent URLs; only cite URLs that appear there.
- If scraped lists are empty, you may use well-known canonical documentation URLs.
- Return ONLY valid JSON — no markdown fences, no commentary.
- Skip or shorten stages that the learner already covered in knownSkills.
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        provider: str = "auto",
    ):
        self.gemini_key = _usable_key(
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        )
        self.anthropic_key = _usable_key(
            anthropic_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        )
        self.provider = (provider or "auto").strip().lower()

    async def generate_intelligence(
        self, learner: LearnerInput, web_resources: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate structured learning path, assessments, capstone, and opportunities.
        """
        prompt = self._build_user_prompt(learner, web_resources)
        last_error: Optional[Exception] = None

        for backend in self._backends():
            try:
                if backend == "gemini":
                    logger.info("Invoking Gemini model: gemini-flash-latest")
                    raw = await call_gemini_latest(self.SYSTEM_PROMPT + "\n\n" + prompt, self.gemini_key)
                else:
                    logger.info("Invoking Claude model: claude-sonnet-4-5")
                    raw = await call_claude(prompt, self.anthropic_key, system=self.SYSTEM_PROMPT)
                return parse_llm_json(raw)
            except Exception as e:
                logger.warning(f"{backend} generation failed: {e}")
                last_error = e

        if last_error:
            logger.error(f"All LLM backends failed ({last_error}). Switching to fallback engine.")
        else:
            logger.info("No LLM API key supplied. Executing multi-domain TechPath fallback engine.")
        return self._fallback_generation(learner, web_resources)

    def _backends(self) -> list[str]:
        provider = self.provider
        ordered: list[str] = []
        if provider in ("gemini", "auto") and self.gemini_key:
            ordered.append("gemini")
        if provider in ("claude", "anthropic", "auto") and self.anthropic_key:
            ordered.append("claude")
        if provider in ("claude", "anthropic") and self.gemini_key and "gemini" not in ordered:
            ordered.append("gemini")
        return ordered

    def _build_user_prompt(
        self, learner: LearnerInput, web_resources: Optional[Dict[str, Any]]
    ) -> str:
        trimmed = _trim_web_context(web_resources)
        return f"""
Analyze this learner profile and generate their structured TechPath.

Learner profile:
{json.dumps({
    "goal": learner.goal,
    "currentLevel": learner.currentLevel,
    "knownSkills": learner.knownSkills,
    "hoursPerWeek": learner.hoursPerWeek,
    "location": learner.location,
    "interests": learner.interests,
    "learningPreferences": learner.learningPreferences,
    "depth": learner.depth,
})}

Search results found on the web:
{json.dumps(trimmed.get("search_results", []))}

YouTube videos found:
{json.dumps(trimmed.get("video_results", []))}

Generate the complete JSON response matching the required structure.
Ensure topics and projects match the role '{learner.goal}' specifically.
"""

    def _fallback_generation(self, learner: LearnerInput, web_resources: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Multi-domain deterministic generator matching specific career tracks.
        """
        goal_lower = learner.goal.lower()
        interests = learner.interests or ["videography"]
        niche = interests[0] if interests else "Media Production"
        location = learner.location or "Nigeria"

        # Check domain category
        is_cloud = any(k in goal_lower for k in ["cloud", "devops", "kubernetes", "aws", "infrastructure", "sre", "sysadmin"])
        is_software = any(k in goal_lower for k in ["software", "backend", "frontend", "fullstack", "web", "developer"])
        
        if is_cloud:
            return self._build_cloud_devops_path(learner, niche, location)
        elif is_software:
            return self._build_software_engineer_path(learner, niche, location)
        else:
            return self._build_machine_learning_path(learner, niche, location)

    def _build_cloud_devops_path(self, learner: LearnerInput, niche: str, location: str) -> Dict[str, Any]:
        goal = learner.goal
        return {
            "targetRole": goal,
            "summaryPitch": f"Tailored {goal} roadmap for a {learner.currentLevel} in {location} focusing on {niche}. Designed with strict prerequisite ordering (Linux & Networking Foundations → Containerization → Cloud Infra → CI/CD & Orchestration → Observability → SRE).",
            "skillGaps": [
                {
                    "skill": "Linux Systems Administration & Networking",
                    "category": "Foundation",
                    "isRequiredFor": "Server Administration, Bash Scripting & Cloud Networking",
                    "confidence": 0.95,
                    "explanation": "Essential prerequisite before containerization, VPC networking, and Kubernetes orchestration."
                },
                {
                    "skill": "Docker Containerization",
                    "category": "Core",
                    "isRequiredFor": "Packaging Applications & Microservices Architecture",
                    "confidence": 0.92,
                    "explanation": "Critical prerequisite before cluster management with Kubernetes and Helm."
                },
                {
                    "skill": "Infrastructure as Code (Terraform)",
                    "category": "Specialization",
                    "isRequiredFor": "Automated Cloud Provisioning & VPC Management",
                    "confidence": 0.88,
                    "explanation": "Required for reproducible cloud infrastructure management across AWS, Azure, or GCP."
                }
            ],
            "roadmap": [
                {
                    "stageNumber": 1,
                    "stageName": "1. Foundations",
                    "description": "Master storyboarding, script writing, camera positioning, and audio capture.",
                    "focusTopics": ["Storytelling Mechanics", "Scriptwriting & Hooks", "Lighting & Framing", "Audio Capture"],
                    "prerequisitesRequired": ["None"],
                    "resources": [
                        {
                            "title": "YouTube Creator Academy - Video Production Basics",
                            "url": "https://creatoracademy.youtube.com/",
                            "resourceType": "course",
                            "difficulty": "Beginner",
                            "estimatedHours": 8,
                            "skillsTaught": ["Scriptwriting", "Lighting", "Audio"],
                            "prerequisites": [],
                            "summary": "Official beginner guide to structuring engaging video content.",
                            "isFree": True
                        },
                        {
                            "title": "Smartphone Videography & Lighting Masterclass",
                            "url": "https://www.youtube.com/",
                            "resourceType": "youtube",
                            "difficulty": "Beginner",
                            "estimatedHours": 3,
                            "skillsTaught": ["Mobile Shooting", "Lighting Setup"],
                            "prerequisites": [],
                            "summary": "Practical tutorial on capturing high quality video with accessible gear.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-1",
                            "stage": "1. Foundations",
                            "questionType": "conceptual",
                            "question": "What is the 3-second hook rule in video production? Why is crisp audio more critical to viewer retention than 4K video resolution?",
                            "options": None,
                            "correctAnswerOrRubric": "The 3-second hook establishes immediate value/curiosity. Viewers tolerate low video quality but instantly click off poor audio.",
                            "explanation": "Tests core understanding of audience retention and production priorities."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-1",
                            "title": "60-Second Short-Form Script & Video Teaser",
                            "stage": "1. Foundations",
                            "projectType": "mini_project",
                            "problemStatement": "Write a 60-second script for a video in your niche (" + niche + "), film it with clean lighting/audio, and produce a short clip.",
                            "objectives": ["Write a strong hook", "Film with clear audio", "Apply rule-of-thirds framing"],
                            "skillsTested": ["Scriptwriting", "Framing", "Audio"],
                            "suggestedStack": ["Smartphone/Camera", "CapCut / Premiere", "Mic"],
                            "deliverables": ["60-second video MP4", "Written script document"],
                            "evaluationRubric": ["Hook clarity in first 3s", "Audio noise floor", "Framing quality"]
                        }
                    ]
                },
                {
                    "stageNumber": 2,
                    "stageName": "2. Core Skills",
                    "description": "Master non-linear video editing software (Premiere Pro, DaVinci Resolve, CapCut Pro) and audio mixing.",
                    "focusTopics": ["Timeline Editing & Pacing", "Color Grading & LUTs", "Audio Noise Reduction", "Motion Graphics"],
                    "prerequisitesRequired": ["Storyboarding", "Basic Footage"],
                    "resources": [
                        {
                            "title": "DaVinci Resolve / Premiere Pro Editing Complete Guide",
                            "url": "https://www.blackmagicdesign.com/products/davinciresolve/training",
                            "resourceType": "documentation",
                            "difficulty": "Intermediate",
                            "estimatedHours": 12,
                            "skillsTaught": ["Video Editing", "Color Grading", "Fairlight Audio"],
                            "prerequisites": ["Footage Capture"],
                            "summary": "Full post-production workflow training.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-2",
                            "stage": "2. Core Skills",
                            "questionType": "conceptual",
                            "question": "Explain J-cuts and L-cuts in video editing. How do B-roll overlays prevent visual monotony?",
                            "options": None,
                            "correctAnswerOrRubric": "J-cut audio starts before video; L-cut video starts before audio. B-roll visually reinforces spoken topics.",
                            "explanation": "Verifies editing pacing and visual storytelling skills."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-2",
                            "title": "5-Minute Edited Vlog / Tutorial with B-Roll",
                            "stage": "2. Core Skills",
                            "projectType": "integration_project",
                            "problemStatement": "Edit a 5-minute structured video with title cards, color grading, background music ducking, and relevant B-roll clips.",
                            "objectives": ["Master J/L cuts", "Apply audio ducking", "Grade color contrast"],
                            "skillsTested": ["Video Editing", "Audio Mixing", "B-Roll Placement"],
                            "suggestedStack": ["DaVinci Resolve", "Premiere Pro", "CapCut"],
                            "deliverables": ["Exported 1080p Video", "Editing project file"],
                            "evaluationRubric": ["Seamless audio transitions", "Pacing & engagement", "Clean color balance"]
                        }
                    ]
                },
                {
                    "stageNumber": 3,
                    "stageName": "3. Specialization",
                    "description": "Develop a distinct visual brand and editing style in " + niche + ".",
                    "focusTopics": [niche + " Techniques", "Cinematic Transitions", "Custom Sound Design", "Brand Aesthetics"],
                    "prerequisitesRequired": ["Video Editing Core", "Scriptwriting"],
                    "resources": [
                        {
                            "title": "Cinematography & Visual Storytelling Course",
                            "url": "https://www.skillshare.com/",
                            "resourceType": "course",
                            "difficulty": "Intermediate",
                            "estimatedHours": 10,
                            "skillsTaught": ["Cinematography", "Lighting Design", "Niche Aesthetics"],
                            "prerequisites": ["Basic Editing"],
                            "summary": "Specialized techniques for crafting visual identity.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-3",
                            "stage": "3. Specialization",
                            "questionType": "conceptual",
                            "question": "How do lens focal length, aperture (depth of field), and shutter angle affect the cinematic feel of " + niche + " footage?",
                            "options": None,
                            "correctAnswerOrRubric": "180-degree shutter angle gives natural motion blur; shallow depth of field separates subject from background.",
                            "explanation": "Validates technical camera operation and camera physics."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-3",
                            "title": niche + " Signature Brand Showcase Video",
                            "stage": "3. Specialization",
                            "projectType": "portfolio_project",
                            "problemStatement": "Create a high-production 3-minute brand showcase video specifically highlighting " + niche + " techniques.",
                            "objectives": ["Execute cinematic lighting", "Design custom intro/outro", "Implement sound design"],
                            "skillsTested": ["Cinematography", "Sound Design", "Niche Production"],
                            "suggestedStack": ["Mirrorless/DSLR or Pro Phone", "NLE", "Audition"],
                            "deliverables": ["Master 4K/1080p Video", "Thumbnail Options"],
                            "evaluationRubric": ["Visual uniqueness", "Sound design richness", "Niche relevance"]
                        }
                    ]
                },
                {
                    "stageNumber": 4,
                    "stageName": "4. Build & Produce",
                    "description": "Establish content batching, thumbnail design, YouTube SEO, and multi-platform publishing workflows.",
                    "focusTopics": ["Click-Through Rate (CTR) Design", "YouTube/TikTok SEO", "Content Batching", "Repurposing Workflows"],
                    "prerequisitesRequired": ["Editing", "Specialization"],
                    "resources": [
                        {
                            "title": "Thumbnail & Graphic Design for Creators (Canva / Photoshop)",
                            "url": "https://www.adobe.com/",
                            "resourceType": "documentation",
                            "difficulty": "Intermediate",
                            "estimatedHours": 6,
                            "skillsTaught": ["Thumbnail Design", "CTR Optimization", "Graphic Layout"],
                            "prerequisites": ["Basic Design"],
                            "summary": "Guide to designing eye-catching thumbnails and graphics.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-4",
                            "stage": "4. Build & Produce",
                            "questionType": "conceptual",
                            "question": "Why should thumbnails complement title text rather than repeat it? What is A/B testing in thumbnail optimization?",
                            "options": None,
                            "correctAnswerOrRubric": "Thumbnail and title form a combined 2-part hook. A/B testing measures which thumbnail generates higher Click-Through Rate.",
                            "explanation": "Tests publishing psychology and click-through optimization."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-4",
                            "title": "Multi-Platform Content Launch Campaign",
                            "stage": "4. Build & Produce",
                            "projectType": "integration_project",
                            "problemStatement": "Produce 1 main long-form video, batch 3 short-form clips, and design 2 high-CTR thumbnail variants.",
                            "objectives": ["Batch production", "Design high-CTR thumbnails", "Repurpose long-form to Reels/Shorts"],
                            "skillsTested": ["Canva/Photoshop", "Shorts Editing", "SEO Metadata"],
                            "suggestedStack": ["Canva", "Photoshop", "CapCut", "YouTube Studio"],
                            "deliverables": ["Long-form video", "3 Shorts clips", "2 Thumbnail PNGs", "Metadata sheet"],
                            "evaluationRubric": ["Thumbnail contrast & readability", "Repurposing efficiency", "SEO title hook"]
                        }
                    ]
                },
                {
                    "stageNumber": 5,
                    "stageName": "5. Prove & Publish",
                    "description": "Launch your public channel, participate in creator challenges, and collaborate with peers.",
                    "focusTopics": ["Channel Launch", "Creator Collaborations", "Community Engagement", "Brand Kit"],
                    "prerequisitesRequired": ["Batch Content", "Thumbnails"],
                    "resources": [
                        {
                            "title": "YouTube Creator Growth & Channel Monetization Blueprint",
                            "url": "https://youtube.com/creators",
                            "resourceType": "documentation",
                            "difficulty": "Intermediate",
                            "estimatedHours": 5,
                            "skillsTaught": ["Community Building", "Analytics", "Monetization Requirements"],
                            "prerequisites": ["Channel Setup"],
                            "summary": "Official strategy for growing a subscriber base and building community.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-5",
                            "stage": "5. Prove & Publish",
                            "questionType": "conceptual",
                            "question": "How do Average Percentage Viewed (APV) and Audience Retention graphs dictate video placement in recommendation algorithms?",
                            "options": None,
                            "correctAnswerOrRubric": "Higher retention signals video satisfaction, triggering algorithmic recommendations to wider audiences.",
                            "explanation": "Evaluates channel growth analytics interpretation."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-5",
                            "title": "Public Channel Launch & Creator Media Kit",
                            "stage": "5. Prove & Publish",
                            "projectType": "portfolio_project",
                            "problemStatement": "Publish a 4-video series on a live YouTube/TikTok channel, assemble a media kit, and track retention metrics.",
                            "objectives": ["Publish live series", "Analyze YouTube Analytics", "Create Creator Media Kit"],
                            "skillsTested": ["Channel Management", "Analytics", "Media Kit Design"],
                            "suggestedStack": ["YouTube / TikTok", "Canva", "Notion"],
                            "deliverables": ["Live Channel Link", "PDF Media Kit", "Analytics Review"],
                            "evaluationRubric": ["Consistent uploads", "Professional media kit", "Metric tracking"]
                        }
                    ]
                },
                {
                    "stageNumber": 6,
                    "stageName": "6. Advance & Scale",
                    "description": "Automate editing workflows using AI tools (CapCut AI, Descript, Runway), secure brand sponsorships, and scale production.",
                    "focusTopics": ["AI Video Tools (Runway / Descript)", "Sponsorship Pitching", "Content Automation", "Production Team Scaling"],
                    "prerequisitesRequired": ["Active Channel", "Consistent Uploads"],
                    "resources": [
                        {
                            "title": "AI Tools for Creators & Editors (Descript & Runway)",
                            "url": "https://www.descript.com/",
                            "resourceType": "course",
                            "difficulty": "Advanced",
                            "estimatedHours": 6,
                            "skillsTaught": ["AI Editing", "Text-Based Editing", "Voice Cloning"],
                            "prerequisites": ["Video Editing Core"],
                            "summary": "Modern AI workflows for accelerating video production.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-6",
                            "stage": "6. Advance & Scale",
                            "questionType": "conceptual",
                            "question": "How do text-based video editors (Descript) and AI generative tools (Runway) streamline post-production workflows?",
                            "options": None,
                            "correctAnswerOrRubric": "Text-based editing lets you edit video by editing transcript text; generative AI produces synthetic B-roll.",
                            "explanation": "Assesses modern AI media production capabilities."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-6",
                            "title": "AI-Augmented Documentary & Brand Sponsorship Pitch",
                            "stage": "6. Advance & Scale",
                            "projectType": "portfolio_project",
                            "problemStatement": "Produce a mini-documentary leveraging AI tools for script polishing, auto-subtitles, and generative B-roll, alongside a brand pitch deck.",
                            "objectives": ["Integrate AI tools", "Pitch brand sponsors", "Scale production output"],
                            "skillsTested": ["AI Video Generation", "Brand Pitching", "Production Scaling"],
                            "suggestedStack": ["Descript", "Runway Gen-2", "DaVinci", "Pitch Deck"],
                            "deliverables": ["Mini-Doc Video", "Sponsorship Proposal Deck"],
                            "evaluationRubric": ["AI integration quality", "Sponsorship proposal value", "Production polish"]
                        }
                    ]
                }
            ],
            "capstone": {
                "title": f"High-Impact {niche} Digital Media Campaign & Video Series",
                "domainTarget": f"{goal} - {niche}",
                "problemStatement": f"Plan, film, edit, and launch a complete 3-part video series focusing on {niche}. The campaign includes brand guidelines, thumbnail A/B testing, AI-assisted post-production, and a sponsor pitch proposal.",
                "objectives": [
                    f"Curate and script 3 original videos in {niche}",
                    "Film with professional lighting, audio, and camera framing",
                    "Edit with custom color grading, sound design, and motion graphics",
                    "Publish with optimized thumbnails and track audience analytics"
                ],
                "skillsTested": ["Scriptwriting", "Videography", "Editing", "Canva/Photoshop", "YouTube SEO", "AI Tools"],
                "suggestedStack": ["DaVinci Resolve / Premiere", "CapCut", "Descript", "Canva", "YouTube Studio"],
                "deliverables": [
                    "3 Published High-Quality Videos",
                    "2 Custom Thumbnails per video",
                    "Creator Media Kit & Sponsorship Deck",
                    "Analytics & Audience Retention Report"
                ],
                "milestones": [
                    "Milestone 1: Scriptwriting, storyboarding, and gear setup",
                    "Milestone 2: Filming, audio capture, and raw footage organization",
                    "Milestone 3: Video editing, color grading, and sound design",
                    "Milestone 4: Publishing, thumbnail design, and campaign launch"
                ],
                "evaluationCriteria": [
                    "Viewer retention (>50% average retention target)",
                    "Thumbnail Click-Through Rate (CTR > 6%)",
                    "Audio and visual production quality",
                    "Consistency of visual branding"
                ],
                "extensionIdeas": [
                    "Create automated short-form clips using CapCut AI for Instagram & TikTok",
                    "Launch a newsletter or community discord for video subscribers"
                ]
            },
            "opportunities": [
                {
                    "title": "Cloud Native Computing Foundation (CNCF) Community Challenge",
                    "organizer": "CNCF Foundation",
                    "opportunityType": "hackathon",
                    "deadline": "2026-11-15",
                    "eligibility": "Open to cloud developers & DevOps engineers globally",
                    "remoteStatus": "Remote",
                    "requiredSkills": ["Kubernetes", "Docker", "Cloud Native", "DevOps"],
                    "difficultyEstimate": "Intermediate",
                    "applicationUrl": "https://cncf.io/",
                    "sourceUrl": "https://cncf.io/",
                    "matchReason": f"Directly aligns with your target role ({goal}) and cloud engineering goals."
                },
                {
                    "title": "AWS Community Builders & Hackathon",
                    "organizer": "Amazon Web Services",
                    "opportunityType": "competition",
                    "deadline": "2026-12-01",
                    "eligibility": "Cloud architects and builders",
                    "remoteStatus": "Remote",
                    "requiredSkills": ["AWS", "Terraform", "Cloud Architecture"],
                    "difficultyEstimate": "Intermediate to Advanced",
                    "applicationUrl": "https://aws.amazon.com/developer/community/builders/",
                    "sourceUrl": "https://aws.amazon.com/",
                    "matchReason": f"Great opportunity to build AWS cloud solutions and network with cloud engineers."
                }
            ],
            "sources": [
                "https://linuxjourney.com/",
                "https://docs.docker.com/",
                "https://developer.hashicorp.com/terraform",
                "https://kubernetes.io/docs/",
                "https://prometheus.io/docs/"
            ]
        }

    def _build_software_engineer_path(self, learner: LearnerInput, niche: str, location: str) -> Dict[str, Any]:
        goal = learner.goal
        return {
            "targetRole": goal,
            "summaryPitch": f"Tailored {goal} roadmap for a {learner.currentLevel} in {location} focusing on {niche}. Designed with strict prerequisite ordering (Foundations → Core → Specialization → Build → Prove → Advance).",
            "skillGaps": [
                {
                    "skill": "Data Structures & Algorithms",
                    "category": "Foundation",
                    "isRequiredFor": "System Efficiency & Coding Interviews",
                    "confidence": 0.95,
                    "explanation": "Essential foundation for writing scalable algorithms and solving complex problems."
                },
                {
                    "skill": "Database Architecture & SQL/NoSQL",
                    "category": "Core",
                    "isRequiredFor": "Backend Data Persistence",
                    "confidence": 0.90,
                    "explanation": "Prerequisite for building data-backed application services."
                }
            ],
            "roadmap": [
                {
                    "stageNumber": 1,
                    "stageName": "1. Foundations",
                    "description": "Master core programming, version control, and computer science basics.",
                    "focusTopics": ["Python / JavaScript", "Git & GitHub", "Object-Oriented Programming", "Command Line"],
                    "prerequisitesRequired": ["None"],
                    "resources": [
                        {
                            "title": "CS50: Introduction to Computer Science",
                            "url": "https://cs50.harvard.edu/",
                            "resourceType": "course",
                            "difficulty": "Beginner",
                            "estimatedHours": 20,
                            "skillsTaught": ["C", "Python", "Algorithms", "Data Structures"],
                            "prerequisites": [],
                            "summary": "Gold standard introduction to computer science and programming fundamentals.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-1",
                            "stage": "1. Foundations",
                            "questionType": "coding_prompt",
                            "question": "Implement a stack data structure with push, pop, and peek operations in Python/JavaScript. Explain Big-O time complexity.",
                            "options": None,
                            "correctAnswerOrRubric": "Stack implemented using list/array with O(1) push and pop.",
                            "explanation": "Tests fundamental data structure concepts."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-1",
                            "title": "CLI Task Manager & File Processor",
                            "stage": "1. Foundations",
                            "projectType": "mini_project",
                            "problemStatement": "Build a command line application that manages tasks and saves state to JSON files.",
                            "objectives": ["File I/O", "Data structures", "OOP design"],
                            "skillsTested": ["Python", "JSON", "CLI"],
                            "suggestedStack": ["Python 3.11", "Argparse"],
                            "deliverables": ["CLI codebase", "README"],
                            "evaluationRubric": ["Error handling", "Clean OOP structure"]
                        }
                    ]
                },
                {
                    "stageNumber": 2,
                    "stageName": "2. Core Skills",
                    "description": "Build relational databases, SQL queries, and RESTful API endpoints.",
                    "focusTopics": ["SQL & PostgreSQL", "REST API Design", "Data Structures", "HTTP Protocols"],
                    "prerequisitesRequired": ["Programming Foundations"],
                    "resources": [
                        {
                            "title": "PostgreSQL & SQL Mastery Guide",
                            "url": "https://www.postgresql.org/docs/",
                            "resourceType": "documentation",
                            "difficulty": "Intermediate",
                            "estimatedHours": 10,
                            "skillsTaught": ["SQL", "Relational Database Design", "Indexing"],
                            "prerequisites": ["Basic Programming"],
                            "summary": "Complete guide to SQL databases.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-2",
                            "stage": "2. Core Skills",
                            "questionType": "conceptual",
                            "question": "Explain database normalization (1NF, 2NF, 3NF) and the difference between INNER JOIN and LEFT JOIN.",
                            "options": None,
                            "correctAnswerOrRubric": "Normalization eliminates redundancy. INNER JOIN returns matching rows; LEFT JOIN returns all left rows.",
                            "explanation": "Verifies SQL and database architecture knowledge."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-2",
                            "title": "Database-Backed RESTful Microservice",
                            "stage": "2. Core Skills",
                            "projectType": "integration_project",
                            "problemStatement": "Develop a REST API with CRUD endpoints backed by a PostgreSQL database.",
                            "objectives": ["Database schema design", "SQL integration", "REST endpoints"],
                            "skillsTested": ["Python/Node", "SQL", "FastAPI/Express"],
                            "suggestedStack": ["FastAPI", "PostgreSQL", "SQLAlchemy"],
                            "deliverables": ["API Codebase", "SQL Migration Scripts"],
                            "evaluationRubric": ["Clean CRUD endpoints", "SQL query safety"]
                        }
                    ]
                },
                {
                    "stageNumber": 3,
                    "stageName": "3. Specialization",
                    "description": "Specialize in " + niche + " backend architecture, microservices, and system design.",
                    "focusTopics": [niche + " Microservices", "Authentication (JWT/OAuth)", "Caching (Redis)", "Async Queues"],
                    "prerequisitesRequired": ["REST APIs", "SQL"],
                    "resources": [
                        {
                            "title": "System Design Primer - GitHub",
                            "url": "https://github.com/donnemartin/system-design-primer",
                            "resourceType": "repository",
                            "difficulty": "Intermediate",
                            "estimatedHours": 25,
                            "skillsTaught": ["System Design", "Scalability", "Caching", "Load Balancing"],
                            "prerequisites": ["Core Web Dev"],
                            "summary": "Comprehensive guide to designing large-scale systems.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-3",
                            "stage": "3. Specialization",
                            "questionType": "conceptual",
                            "question": "How does Redis caching improve API response times? What is cache invalidation?",
                            "options": None,
                            "correctAnswerOrRubric": "Redis stores key-value pairs in memory for sub-millisecond retrieval. Invalidation removes stale cache items.",
                            "explanation": "Tests backend caching and scalability."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-3",
                            "title": niche + " High-Throughput Microservice",
                            "stage": "3. Specialization",
                            "projectType": "portfolio_project",
                            "problemStatement": f"Build a resilient microservice designed for {niche} with JWT authentication and Redis caching.",
                            "objectives": ["Implement JWT auth", "Cache frequent queries", "Handle rate limiting"],
                            "skillsTested": ["FastAPI", "Redis", "PostgreSQL", "JWT"],
                            "suggestedStack": ["FastAPI", "Redis", "PostgreSQL"],
                            "deliverables": ["Microservice Repo", "Swagger API Docs"],
                            "evaluationRubric": ["Sub-100ms response time", "Secure auth logic"]
                        }
                    ]
                },
                {
                    "stageNumber": 4,
                    "stageName": "4. Build & Deploy",
                    "description": "Containerize services with Docker and deploy to cloud platforms with CI/CD pipelines.",
                    "focusTopics": ["Docker & Docker Compose", "CI/CD Pipelines (GitHub Actions)", "Cloud Deployment (AWS/Render)", "Monitoring"],
                    "prerequisitesRequired": ["Microservices", "Git"],
                    "resources": [
                        {
                            "title": "Docker & Kubernetes Official Guides",
                            "url": "https://docs.docker.com/",
                            "resourceType": "documentation",
                            "difficulty": "Intermediate",
                            "estimatedHours": 10,
                            "skillsTaught": ["Docker", "Containerization", "Compose"],
                            "prerequisites": ["Command Line"],
                            "summary": "Containerization standards for software applications.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-4",
                            "stage": "4. Build & Deploy",
                            "questionType": "conceptual",
                            "question": "Explain the difference between a Docker image and a Docker container. How does multi-stage building reduce container image size?",
                            "options": None,
                            "correctAnswerOrRubric": "Image is a static template; container is a running instance. Multi-stage builds strip out build-time dependencies.",
                            "explanation": "Validates containerization efficiency."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-4",
                            "title": "Containerized CI/CD Production Deployment",
                            "stage": "4. Build & Deploy",
                            "projectType": "integration_project",
                            "problemStatement": "Setup docker-compose for app + DB + Redis, and configure GitHub Actions for automated testing and deployment.",
                            "objectives": ["Multi-container compose", "Automate CI/CD", "Deploy to cloud"],
                            "skillsTested": ["Docker Compose", "GitHub Actions", "Cloud Hosting"],
                            "suggestedStack": ["Docker", "GitHub Actions", "Render / AWS"],
                            "deliverables": ["Dockerfile", "docker-compose.yml", "Live App URL"],
                            "evaluationRubric": ["Automated test pass on push", "Zero-downtime deploy"]
                        }
                    ]
                },
                {
                    "stageNumber": 5,
                    "stageName": "5. Prove & Publish",
                    "description": "Publish open-source tools, contribute to active GitHub projects, and publish Apify Actors.",
                    "focusTopics": ["Apify Actor Publishing", "Open Source Contribution", "Tech Writing"],
                    "prerequisitesRequired": ["Docker", "Git", "REST APIs"],
                    "resources": [
                        {
                            "title": "Apify Open Source Actor Development Guide",
                            "url": "https://apify.com/store",
                            "resourceType": "documentation",
                            "difficulty": "Intermediate",
                            "estimatedHours": 5,
                            "skillsTaught": ["Apify SDK", "Actor Publishing"],
                            "prerequisites": ["Python / JS"],
                            "summary": "Guide on publishing serverless cloud tools.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-5",
                            "stage": "5. Prove & Publish",
                            "questionType": "coding_prompt",
                            "question": "Draft an Open Source PR description detailing a bug fix, root cause analysis, and unit test verification.",
                            "options": None,
                            "correctAnswerOrRubric": "Clear title, root cause, code fix details, and test logs.",
                            "explanation": "Assesses professional collaboration skills."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-5",
                            "title": "Apify Cloud Automation Actor",
                            "stage": "5. Prove & Publish",
                            "projectType": "portfolio_project",
                            "problemStatement": "Develop and publish a serverless Apify Actor that automates web data extraction and provides API endpoints.",
                            "objectives": ["Define input schema", "Process data", "Publish to Apify Store"],
                            "skillsTested": ["Apify SDK", "Python/Node", "API Design"],
                            "suggestedStack": ["Apify SDK", "Docker", "Pydantic"],
                            "deliverables": ["Published Apify Actor Link", "Public GitHub Repo"],
                            "evaluationRubric": ["Clean input/output schema", "Published on Store"]
                        }
                    ]
                },
                {
                    "stageNumber": 6,
                    "stageName": "6. Advance & Scale",
                    "description": "Master distributed systems, event-driven architecture (Kafka/RabbitMQ), and cloud architecture.",
                    "focusTopics": ["Event-Driven Architecture (Kafka)", "Microservices Resiliency", "Kubernetes", "Observability"],
                    "prerequisitesRequired": ["Stages 1 to 5 satisfied"],
                    "resources": [
                        {
                            "title": "Designing Data-Intensive Applications",
                            "url": "https://dataintensive.net/",
                            "resourceType": "book",
                            "difficulty": "Advanced",
                            "estimatedHours": 30,
                            "skillsTaught": ["Distributed Systems", "Replication", "Partitioning", "Transactions"],
                            "prerequisites": ["Backend Core"],
                            "summary": "Definitive guide to system architecture and distributed data.",
                            "isFree": False
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-6",
                            "stage": "6. Advance & Scale",
                            "questionType": "conceptual",
                            "question": "Compare message queues (RabbitMQ/Kafka) with synchronous HTTP REST calls for inter-service communication.",
                            "options": None,
                            "correctAnswerOrRubric": "Message queues decouple producers and consumers asynchronously, preventing cascading failures.",
                            "explanation": "Evaluates distributed systems architecture comprehension."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-6",
                            "title": f"Distributed Event-Driven {niche} Platform",
                            "stage": "6. Advance & Scale",
                            "projectType": "portfolio_project",
                            "problemStatement": f"Build a multi-service event-driven architecture for {niche} using Kafka, PostgreSQL, and distributed tracing.",
                            "objectives": ["Event streaming with Kafka", "Distributed logging", "High availability"],
                            "skillsTested": ["Kafka", "Docker", "Microservices", "Monitoring"],
                            "suggestedStack": ["Kafka", "Python/Go", "PostgreSQL", "Prometheus"],
                            "deliverables": ["Multi-repo architecture", "System benchmark report"],
                            "evaluationRubric": ["Fault tolerance under load", "Clean event schemas"]
                        }
                    ]
                }
            ],
            "capstone": {
                "title": f"Production-Grade {niche} Microservice & Platform Architecture",
                "domainTarget": f"{goal} - {niche}",
                "problemStatement": f"Develop an end-to-end, deployed backend platform designed for {niche}. The system handles user auth, database persistence, caching, containerization, and cloud API deployment.",
                "objectives": [
                    f"Design relational database schemas for {niche}",
                    "Build high-throughput REST APIs with authentication & caching",
                    "Containerize with Docker Compose and set up GitHub Actions CI/CD",
                    "Deploy live to cloud hosting with API documentation"
                ],
                "skillsTested": ["Python", "SQL", "FastAPI", "Docker", "Redis", "CI/CD"],
                "suggestedStack": ["Python 3.11", "FastAPI", "PostgreSQL", "Redis", "Docker", "GitHub Actions"],
                "deliverables": [
                    "Full Git Repository with clean commits",
                    "Dockerfile & docker-compose.yml setup",
                    "Live deployed API documentation (Swagger)",
                    "System architecture document & load test report"
                ],
                "milestones": [
                    "Milestone 1: Database schema design & ORM models",
                    "Milestone 2: REST API endpoints & JWT authentication",
                    "Milestone 3: Redis caching layer & performance optimization",
                    "Milestone 4: Docker containerization & cloud CI/CD deployment"
                ],
                "evaluationCriteria": [
                    "API endpoint p95 latency under 100ms",
                    "Database query efficiency (indexed foreign keys)",
                    "Clean code modularity and unit test coverage (>80%)",
                    "Comprehensive API documentation"
                ],
                "extensionIdeas": [
                    "Integrate Kafka or RabbitMQ event stream for async background tasks",
                    "Add Prometheus & Grafana dashboard for live server metrics"
                ]
            },
            "opportunities": [
                {
                    "title": "Apify × She Code Africa Hackathon 2026",
                    "organizer": "Apify & She Code Africa",
                    "opportunityType": "hackathon",
                    "deadline": "2026-10-15",
                    "eligibility": "Open to developers and tech learners across Africa",
                    "remoteStatus": "Remote",
                    "requiredSkills": ["Python", "API Development", "Web Crawling", "Docker"],
                    "difficultyEstimate": "Beginner to Intermediate friendly",
                    "applicationUrl": "https://apify.com/hackathons",
                    "sourceUrl": "https://apify.com/",
                    "matchReason": f"Directly aligns with your goal ({goal}). Offers mentorship, cash prizes, and Apify Actor publishing exposure."
                },
                {
                    "title": "Global Open Source Software Fellowship",
                    "organizer": "Global Tech Talent Network",
                    "opportunityType": "internship",
                    "deadline": "2026-11-15",
                    "eligibility": "Early-career developers",
                    "remoteStatus": "Remote",
                    "requiredSkills": ["Git", "Python/JS", "REST APIs"],
                    "difficultyEstimate": "Intermediate",
                    "applicationUrl": "https://shecodeafrica.org/",
                    "sourceUrl": "https://shecodeafrica.org/",
                    "matchReason": f"Provides mentorship bridge into full-time remote engineering roles."
                }
            ],
            "sources": [
                "https://cs50.harvard.edu/",
                "https://www.postgresql.org/",
                "https://fastapi.tiangolo.com/",
                "https://apify.com/store"
            ]
        }

    def _build_machine_learning_path(self, learner: LearnerInput, niche: str, location: str) -> Dict[str, Any]:
        goal = learner.goal
        return {
            "targetRole": goal,
            "summaryPitch": f"Tailored {goal} roadmap for a {learner.currentLevel} in {location} focusing on {niche}. Designed with strict prerequisite ordering (Foundations → Core → Specialization → Build → Prove → Advance).",
            "skillGaps": [
                {
                    "skill": "Linear Algebra & Statistics for ML",
                    "category": "Foundation",
                    "isRequiredFor": "Supervised Learning, Optimization & Model Evaluation",
                    "confidence": 0.90,
                    "explanation": "Prerequisite for understanding model weights, loss functions, and probability."
                },
                {
                    "skill": "Data Structures & Algorithms",
                    "category": "Core",
                    "isRequiredFor": "System Efficiency & Coding Interviews",
                    "confidence": 0.88,
                    "explanation": "Critical for writing scalable algorithms and building production systems."
                }
            ],
            "roadmap": [
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
                            "skillsTaught": ["Python syntax", "Data Structures"],
                            "prerequisites": [],
                            "summary": "Comprehensive beginner introduction to Python programming.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-1",
                            "stage": "1. Foundations",
                            "questionType": "coding_prompt",
                            "question": "Write a Python function `filter_even_squares(numbers)` that takes a list of integers, filters out odd numbers, squares the even numbers, and returns the result in reverse order. Explain time complexity.",
                            "options": None,
                            "correctAnswerOrRubric": "Time complexity O(N), space O(N).",
                            "explanation": "Tests fundamental Python sequence operations."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-1",
                            "title": "Automated Data Processing & Extraction CLI",
                            "stage": "1. Foundations",
                            "projectType": "mini_project",
                            "problemStatement": "Build a command-line script that ingests CSV/JSON dataset files and outputs formatted statistics.",
                            "objectives": ["File I/O", "Pandas", "CLI"],
                            "skillsTested": ["Python", "Pandas"],
                            "suggestedStack": ["Python 3.11", "Pandas"],
                            "deliverables": ["CLI Script", "README"],
                            "evaluationRubric": ["Error handling", "Clean code"]
                        }
                    ]
                },
                {
                    "stageNumber": 2,
                    "stageName": "2. Core Skills",
                    "description": "Build solid fundamentals in Data Structures, Data Analysis, and Supervised Machine Learning.",
                    "focusTopics": ["Data Structures & Algorithms", "SQL Queries", "Supervised Learning", "Scikit-Learn"],
                    "prerequisitesRequired": ["Python", "Basic Math"],
                    "resources": [
                        {
                            "title": "Scikit-Learn Official User Guide",
                            "url": "https://scikit-learn.org/",
                            "resourceType": "documentation",
                            "difficulty": "Intermediate",
                            "estimatedHours": 10,
                            "skillsTaught": ["Scikit-Learn", "Model Training"],
                            "prerequisites": ["Python"],
                            "summary": "Official guide to machine learning algorithms.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-2",
                            "stage": "2. Core Skills",
                            "questionType": "conceptual",
                            "question": "Explain overfitting in machine learning. How do cross-validation and regularization prevent it?",
                            "options": None,
                            "correctAnswerOrRubric": "Overfitting happens when model memorizes training noise.",
                            "explanation": "Verifies core understanding of ML evaluation."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-2",
                            "title": "Supervised Prediction Pipeline",
                            "stage": "2. Core Skills",
                            "projectType": "integration_project",
                            "problemStatement": "Develop an end-to-end classification pipeline that cleans tabular data and evaluates performance.",
                            "objectives": ["Feature engineering", "Model evaluation"],
                            "skillsTested": ["Scikit-Learn", "Pandas"],
                            "suggestedStack": ["Python", "Scikit-Learn"],
                            "deliverables": ["Jupyter notebook", "Trained model pkl"],
                            "evaluationRubric": ["No data leakage", "Proper metrics"]
                        }
                    ]
                },
                {
                    "stageNumber": 3,
                    "stageName": "3. Specialization",
                    "description": "Dive deep into modern Deep Learning, Neural Networks, PyTorch, and " + niche + ".",
                    "focusTopics": ["Neural Networks", "PyTorch", "Computer Vision / NLP", niche],
                    "prerequisitesRequired": ["Core ML", "Linear Algebra"],
                    "resources": [
                        {
                            "title": "Deep Learning Specialization - DeepLearning.AI",
                            "url": "https://www.coursera.org/specializations/deep-learning",
                            "resourceType": "course",
                            "difficulty": "Intermediate",
                            "estimatedHours": 30,
                            "skillsTaught": ["PyTorch", "Neural Nets"],
                            "prerequisites": ["Python"],
                            "summary": "Deep learning architectures and optimization.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-3",
                            "stage": "3. Specialization",
                            "questionType": "coding_prompt",
                            "question": "Implement a custom PyTorch nn.Module for a 3-layer neural network with ReLU and Dropout.",
                            "options": None,
                            "correctAnswerOrRubric": "Defines __init__ and forward pass.",
                            "explanation": "Validates PyTorch hands-on skill."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-3",
                            "title": niche + " ML Model Implementation",
                            "stage": "3. Specialization",
                            "projectType": "portfolio_project",
                            "problemStatement": f"Build a PyTorch model specifically tailored for {niche}.",
                            "objectives": [f"Apply PyTorch to {niche}", "Optimize loss"],
                            "skillsTested": ["PyTorch", "Data Preprocessing"],
                            "suggestedStack": ["PyTorch", "FastAPI"],
                            "deliverables": ["Code repo", "Trained weights"],
                            "evaluationRubric": ["Functional model training", "Clean code"]
                        }
                    ]
                },
                {
                    "stageNumber": 4,
                    "stageName": "4. Build & Deploy",
                    "description": "Transform standalone models into robust production microservices.",
                    "focusTopics": ["FastAPI", "Docker", "API Deployment", "Model Serving"],
                    "prerequisitesRequired": ["PyTorch", "Python"],
                    "resources": [
                        {
                            "title": "Deploying ML Models with FastAPI & Docker",
                            "url": "https://fastapi.tiangolo.com/",
                            "resourceType": "documentation",
                            "difficulty": "Intermediate",
                            "estimatedHours": 8,
                            "skillsTaught": ["FastAPI", "Docker"],
                            "prerequisites": ["Python"],
                            "summary": "Production guide for serving ML model predictions.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-4",
                            "stage": "4. Build & Deploy",
                            "questionType": "conceptual",
                            "question": "What is the difference between batch inference and real-time API inference?",
                            "options": None,
                            "correctAnswerOrRubric": "Real-time serves low-latency requests; batch processes bulk data.",
                            "explanation": "Tests deployment principles."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-4",
                            "title": "Containerized ML Inference Microservice",
                            "stage": "4. Build & Deploy",
                            "projectType": "integration_project",
                            "problemStatement": "Wrap your trained model inside a Dockerized FastAPI app.",
                            "objectives": ["Expose REST API", "Containerize app"],
                            "skillsTested": ["FastAPI", "Docker"],
                            "suggestedStack": ["FastAPI", "Docker"],
                            "deliverables": ["Dockerfile", "API code"],
                            "evaluationRubric": ["Container compiles", "Input validation"]
                        }
                    ]
                },
                {
                    "stageNumber": 5,
                    "stageName": "5. Prove & Publish",
                    "description": "Validate your skills in real-world environments through hackathons and Apify Actors.",
                    "focusTopics": ["Hackathons", "Apify Actor Publishing", "Open Source"],
                    "prerequisitesRequired": ["Model Training", "REST APIs"],
                    "resources": [
                        {
                            "title": "Apify Open Source Actor Development Guide",
                            "url": "https://apify.com/store",
                            "resourceType": "documentation",
                            "difficulty": "Intermediate",
                            "estimatedHours": 5,
                            "skillsTaught": ["Apify SDK", "Actor Publishing"],
                            "prerequisites": ["Python"],
                            "summary": "Guide on publishing serverless cloud tools.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-5",
                            "stage": "5. Prove & Publish",
                            "questionType": "coding_prompt",
                            "question": "Draft an Open Source PR description detailing a bug fix and test results.",
                            "options": None,
                            "correctAnswerOrRubric": "Clear title, problem description, fix details.",
                            "explanation": "Evaluates open-source communication."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-5",
                            "title": "Apify Open Source Cloud Tool / Actor",
                            "stage": "5. Prove & Publish",
                            "projectType": "portfolio_project",
                            "problemStatement": "Publish a reusable Apify Actor that automates web data extraction.",
                            "objectives": ["Create input schema", "Publish to Apify Store"],
                            "skillsTested": ["Apify SDK", "Python"],
                            "suggestedStack": ["Apify SDK", "Docker"],
                            "deliverables": ["Published Apify Actor"],
                            "evaluationRubric": ["Valid input/output schema"]
                        }
                    ]
                },
                {
                    "stageNumber": 6,
                    "stageName": "6. Advance (LLMs & Agents)",
                    "description": "Expand into cutting-edge architectures: LLMs, RAG Systems, AI Agents, and MLOps.",
                    "focusTopics": ["RAG Systems", "AI Agents", "MLOps"],
                    "prerequisitesRequired": ["Stages 1 to 5 satisfied"],
                    "resources": [
                        {
                            "title": "Full Stack LLM & RAG Application Architecture",
                            "url": "https://www.deeplearning.ai/",
                            "resourceType": "course",
                            "difficulty": "Advanced",
                            "estimatedHours": 12,
                            "skillsTaught": ["RAG", "Vector Search"],
                            "prerequisites": ["PyTorch"],
                            "summary": "Building blocks for AI agents and RAG applications.",
                            "isFree": True
                        }
                    ],
                    "assessments": [
                        {
                            "id": "quiz-stage-6",
                            "stage": "6. Advance (LLMs & Agents)",
                            "questionType": "conceptual",
                            "question": "How does Retrieval-Augmented Generation (RAG) overcome LLM context window limits?",
                            "options": None,
                            "correctAnswerOrRubric": "RAG fetches relevant chunked documents via embedding similarity.",
                            "explanation": "Assesses modern LLM architecture choices."
                        }
                    ],
                    "projects": [
                        {
                            "id": "proj-stage-6",
                            "title": "Autonomous AI Agent System with Tool Calling",
                            "stage": "6. Advance (LLMs & Agents)",
                            "projectType": "portfolio_project",
                            "problemStatement": "Develop a multi-tool AI Agent capable of web searching and data processing.",
                            "objectives": ["Implement tool selection", "Manage tasks"],
                            "skillsTested": ["LLM APIs", "Vector Database"],
                            "suggestedStack": ["Python", "Gemini API", "ChromaDB"],
                            "deliverables": ["Agent codebase", "Live demo"],
                            "evaluationRubric": ["Reliable tool execution", "Minimal hallucination"]
                        }
                    ]
                }
            ],
            "capstone": {
                "title": f"Production-Grade {niche} Intelligence & Decision System",
                "domainTarget": f"{goal} - {niche}",
                "problemStatement": f"Develop an end-to-end, deployed AI platform designed for {niche}. The system ingests raw domain data, applies PyTorch models, and exposes a containerized API.",
                "objectives": [
                    f"Curate and preprocess datasets for {niche}",
                    "Train a high-performing PyTorch model",
                    "Deploy containerized FastAPI microservice"
                ],
                "skillsTested": ["Python", "PyTorch", "FastAPI", "Docker"],
                "suggestedStack": ["Python 3.11", "PyTorch", "FastAPI", "Docker"],
                "deliverables": ["Git Repo", "Dockerfile", "Live Demo Link"],
                "milestones": [
                    "Milestone 1: Data acquisition and EDA",
                    "Milestone 2: Baseline vs PyTorch model",
                    "Milestone 3: FastAPI microservice and Docker setup",
                    "Milestone 4: Cloud deployment"
                ],
                "evaluationCriteria": [
                    "Model accuracy/F1-score",
                    "API latency under 200ms"
                ],
                "extensionIdeas": [
                    "Add RAG vector search pipeline"
                ]
            },
            "opportunities": [
                {
                    "title": "Apify × She Code Africa Hackathon 2026",
                    "organizer": "Apify & She Code Africa",
                    "opportunityType": "hackathon",
                    "deadline": "2026-10-15",
                    "eligibility": "Open to developers across Africa",
                    "remoteStatus": "Remote",
                    "requiredSkills": ["Python", "Web Crawling", "API Development"],
                    "difficultyEstimate": "Beginner to Intermediate friendly",
                    "applicationUrl": "https://apify.com/hackathons",
                    "sourceUrl": "https://apify.com/",
                    "matchReason": f"Directly aligns with your goal ({goal})."
                },
                {
                    "title": "Kaggle Community AI Challenge",
                    "organizer": "Kaggle Competitions",
                    "opportunityType": "competition",
                    "deadline": "2026-11-01",
                    "eligibility": "Global community",
                    "remoteStatus": "Remote",
                    "requiredSkills": ["Python", "PyTorch"],
                    "difficultyEstimate": "Intermediate",
                    "applicationUrl": "https://www.kaggle.com/",
                    "sourceUrl": "https://www.kaggle.com/",
                    "matchReason": f"Perfect match for building practical proof in {niche}."
                }
            ],
            "sources": [
                "https://www.py4e.com/",
                "https://scikit-learn.org/",
                "https://coursera.org/",
                "https://fastapi.tiangolo.com/",
                "https://apify.com/store"
            ]
        }
