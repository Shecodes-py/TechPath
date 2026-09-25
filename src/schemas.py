from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


SECRET_INPUT_FIELDS = {
    "geminiApiKey",
    "anthropicApiKey",
    "llmApiKey",
    "apifyToken",
}


class LearnerInput(BaseModel):
    """Input payload accepted by the TechPath Actor."""
    model_config = ConfigDict(extra="ignore")

    goal: str = Field(
        default="Machine Learning Engineer",
        min_length=1,
        description="Target technical career role or objective."
    )
    currentLevel: str = Field(
        default="Beginner",
        description="Current overall level (Beginner, Intermediate, Advanced)."
    )
    knownSkills: List[str] = Field(
        default_factory=lambda: ["Python basics"],
        description="Skills, tools, or languages already mastered by the learner."
    )
    hoursPerWeek: int = Field(
        default=8,
        ge=1,
        le=168,
        description="Dedicated study hours per week."
    )
    location: str = Field(
        default="Nigeria",
        description="Learner location/region for localized opportunity matching."
    )
    interests: List[str] = Field(
        default_factory=lambda: ["NLP", "healthcare AI"],
        description="Specific domains, applications, or subfields of interest."
    )
    learningPreferences: List[str] = Field(
        default_factory=lambda: ["video", "hands-on"],
        description="Preferred media formats (e.g. video, documentation, hands-on projects)."
    )
    depth: str = Field(
        default="deep",
        description="Detail level of generated roadmap ('standard' or 'deep')."
    )
    llmProvider: str = Field(
        default="auto",
        description="LLM backend: auto, gemini, or claude."
    )
    geminiApiKey: Optional[str] = Field(
        default=None,
        description="Optional Google Gemini API key provided directly in input."
    )
    anthropicApiKey: Optional[str] = Field(
        default=None,
        description="Optional Anthropic API key for Claude."
    )
    llmApiKey: Optional[str] = Field(
        default=None,
        description="Generic LLM key used when provider-specific keys are omitted."
    )
    apifyToken: Optional[str] = Field(
        default=None,
        description="Optional Apify API token provided directly in input."
    )
    email: Optional[str] = Field(
        default=None,
        description="Optional email address to receive report via apify/send-email Actor."
    )

    def public_profile(self) -> "LearnerPublicProfile":
        return LearnerPublicProfile.model_validate(
            self.model_dump(exclude=SECRET_INPUT_FIELDS | {"llmProvider"})
        )


class LearnerPublicProfile(BaseModel):
    """Learner fields that are safe to persist in datasets and reports."""
    goal: str
    currentLevel: str
    knownSkills: List[str]
    hoursPerWeek: int
    location: str
    interests: List[str]
    learningPreferences: List[str]
    depth: str


class SkillGap(BaseModel):
    """Represents a missing prerequisite or skill gap."""
    skill: str
    category: str  # Foundation, Core, Specialization, Advanced
    isRequiredFor: str
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    explanation: str


class ResourceItem(BaseModel):
    """Curated learning resource item discovered on the web."""
    title: str
    url: str
    resourceType: str  # course, youtube, documentation, tutorial, repository, book
    difficulty: str  # Beginner, Intermediate, Advanced
    estimatedHours: int
    skillsTaught: List[str]
    prerequisites: List[str]
    summary: str
    isFree: bool = True


class AssessmentQuestion(BaseModel):
    """Stage assessment skill check question or prompt."""
    id: str
    stage: str
    questionType: str  # multiple_choice, conceptual, coding_prompt
    question: str
    options: Optional[List[str]] = None
    correctAnswerOrRubric: str
    explanation: str


class ProjectBrief(BaseModel):
    """Reinforcement or integration project recommendation."""
    id: str
    title: str
    stage: str
    projectType: str  # mini_project, integration_project, portfolio_project
    problemStatement: str
    objectives: List[str]
    skillsTested: List[str]
    suggestedStack: List[str]
    deliverables: List[str]
    evaluationRubric: List[str]


class CapstoneBrief(BaseModel):
    """Portfolio-level capstone project brief aligned with domain interest."""
    title: str
    domainTarget: str
    problemStatement: str
    objectives: List[str]
    skillsTested: List[str]
    suggestedStack: List[str]
    deliverables: List[str]
    milestones: List[str]
    evaluationCriteria: List[str]
    extensionIdeas: List[str]


class OpportunityMatch(BaseModel):
    """Matched real-world growth opportunity (hackathon, internship, competition)."""
    title: str
    organizer: str
    opportunityType: str  # hackathon, competition, internship, open_source, community
    deadline: str
    eligibility: str
    remoteStatus: str  # Remote, Hybrid, On-site
    requiredSkills: List[str]
    difficultyEstimate: str
    applicationUrl: str
    sourceUrl: str
    matchReason: str


class CodingProblem(BaseModel):
    """LeetCode or technical practice problem question."""
    title: str
    difficulty: str  # Easy, Medium, Hard
    url: str
    topic: str
    description: Optional[str] = None


class RoadmapStage(BaseModel):
    """A stage in the dependency-aware learning graph."""
    stageNumber: int
    stageName: str  # 1. Foundations, 2. Core Skills, 3. Specialization, 4. Build, 5. Prove, 6. Advance
    description: str
    focusTopics: List[str]
    prerequisitesRequired: List[str]
    resources: List[ResourceItem]
    assessments: List[AssessmentQuestion]
    projects: List[ProjectBrief]
    codingProblems: Optional[List[CodingProblem]] = Field(default_factory=list)


class TechPathOutput(BaseModel):
    """Complete output dataset produced by the TechPath Actor."""
    profile: LearnerPublicProfile
    targetRole: str
    summaryPitch: str
    skillGaps: List[SkillGap]
    roadmap: List[RoadmapStage]
    capstone: CapstoneBrief
    opportunities: List[OpportunityMatch]
    sources: List[str]
    discoveredVideos: Optional[List[dict]] = Field(default_factory=list)
    discoveredSearchResults: Optional[List[dict]] = Field(default_factory=list)
    generatedAt: str
