from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from core.models import AgentActionType, CoreModel


class GoalType(str, Enum):
    LEARN = "learn"
    RESEARCH = "research"
    IMPLEMENT = "implement"
    COMPARE = "compare"
    TROUBLESHOOT = "troubleshoot"
    EXPLORE = "explore"


class TemplateStep(CoreModel):
    description: str
    action_type: AgentActionType = AgentActionType.NO_OP
    priority: int = Field(default=0, ge=0)
    estimated_tokens: int = Field(default=3000, ge=0)
    tool_hint: str = ""


class PlanTemplate(CoreModel):
    goal_type: GoalType
    steps: list[TemplateStep] = Field(default_factory=list)
    default_max_iterations: int = Field(default=10, ge=1)
    description: str = ""


_LEARN_TEMPLATE = PlanTemplate(
    goal_type=GoalType.LEARN,
    description="Structured learning path from fundamentals to advanced",
    default_max_iterations=12,
    steps=[
        TemplateStep(
            description="Understand the fundamental concepts and definitions",
            action_type=AgentActionType.SEARCH,
            priority=1,
            estimated_tokens=4000,
            tool_hint="search_academic",
        ),
        TemplateStep(
            description="Learn the mathematical foundations and theory",
            action_type=AgentActionType.SEARCH,
            priority=2,
            estimated_tokens=4000,
            tool_hint="search_academic",
        ),
        TemplateStep(
            description="Study practical implementations and examples",
            action_type=AgentActionType.SEARCH,
            priority=3,
            estimated_tokens=3500,
            tool_hint="search_github",
        ),
        TemplateStep(
            description="Explore advanced topics and variants",
            action_type=AgentActionType.SEARCH,
            priority=4,
            estimated_tokens=3500,
            tool_hint="search_academic",
        ),
        TemplateStep(
            description="Synthesize findings into a structured learning path",
            action_type=AgentActionType.GENERATE,
            priority=5,
            estimated_tokens=5000,
            tool_hint="generate_learning_path",
        ),
    ],
)

_RESEARCH_TEMPLATE = PlanTemplate(
    goal_type=GoalType.RESEARCH,
    description="Deep research with source analysis and synthesis",
    default_max_iterations=15,
    steps=[
        TemplateStep(
            description="Define research scope and identify key questions",
            action_type=AgentActionType.PLAN,
            priority=1,
            estimated_tokens=2000,
            tool_hint="",
        ),
        TemplateStep(
            description="Search academic and web sources for relevant material",
            action_type=AgentActionType.SEARCH,
            priority=2,
            estimated_tokens=5000,
            tool_hint="search_academic",
        ),
        TemplateStep(
            description="Read and analyze top sources in depth",
            action_type=AgentActionType.READ,
            priority=3,
            estimated_tokens=6000,
            tool_hint="read_url",
        ),
        TemplateStep(
            description="Rank and evaluate source quality and relevance",
            action_type=AgentActionType.ANALYZE,
            priority=4,
            estimated_tokens=4000,
            tool_hint="rank_sources",
        ),
        TemplateStep(
            description="Synthesize findings into a comprehensive report",
            action_type=AgentActionType.GENERATE,
            priority=5,
            estimated_tokens=5000,
            tool_hint="generate_report",
        ),
    ],
)

_IMPLEMENT_TEMPLATE = PlanTemplate(
    goal_type=GoalType.IMPLEMENT,
    description="Implementation guide with code references and tutorials",
    default_max_iterations=12,
    steps=[
        TemplateStep(
            description="Understand requirements and identify necessary concepts",
            action_type=AgentActionType.SEARCH,
            priority=1,
            estimated_tokens=3000,
            tool_hint="search_web",
        ),
        TemplateStep(
            description="Find implementation examples and code repositories",
            action_type=AgentActionType.SEARCH,
            priority=2,
            estimated_tokens=4000,
            tool_hint="search_github",
        ),
        TemplateStep(
            description="Study documentation and API references",
            action_type=AgentActionType.READ,
            priority=3,
            estimated_tokens=4000,
            tool_hint="read_url",
        ),
        TemplateStep(
            description="Identify best practices and common pitfalls",
            action_type=AgentActionType.SEARCH,
            priority=4,
            estimated_tokens=3000,
            tool_hint="search_web",
        ),
        TemplateStep(
            description="Compile implementation guide with resources",
            action_type=AgentActionType.GENERATE,
            priority=5,
            estimated_tokens=4000,
            tool_hint="generate_report",
        ),
    ],
)

_COMPARE_TEMPLATE = PlanTemplate(
    goal_type=GoalType.COMPARE,
    description="Comparative analysis with structured evaluation",
    default_max_iterations=12,
    steps=[
        TemplateStep(
            description="Identify comparison criteria and evaluation metrics",
            action_type=AgentActionType.PLAN,
            priority=1,
            estimated_tokens=2000,
            tool_hint="",
        ),
        TemplateStep(
            description="Gather information about each option being compared",
            action_type=AgentActionType.SEARCH,
            priority=2,
            estimated_tokens=5000,
            tool_hint="search_web",
        ),
        TemplateStep(
            description="Analyze strengths and weaknesses of each option",
            action_type=AgentActionType.ANALYZE,
            priority=3,
            estimated_tokens=4000,
            tool_hint="compare_sources",
        ),
        TemplateStep(
            description="Create structured comparison with recommendations",
            action_type=AgentActionType.GENERATE,
            priority=4,
            estimated_tokens=4000,
            tool_hint="generate_report",
        ),
    ],
)

_TROUBLESHOOT_TEMPLATE = PlanTemplate(
    goal_type=GoalType.TROUBLESHOOT,
    description="Problem diagnosis and solution finding",
    default_max_iterations=10,
    steps=[
        TemplateStep(
            description="Identify and clearly define the problem",
            action_type=AgentActionType.PLAN,
            priority=1,
            estimated_tokens=2000,
            tool_hint="",
        ),
        TemplateStep(
            description="Search for known issues and their causes",
            action_type=AgentActionType.SEARCH,
            priority=2,
            estimated_tokens=4000,
            tool_hint="search_web",
        ),
        TemplateStep(
            description="Find documented solutions and workarounds",
            action_type=AgentActionType.SEARCH,
            priority=3,
            estimated_tokens=4000,
            tool_hint="search_web",
        ),
        TemplateStep(
            description="Compile solution guide with verification steps",
            action_type=AgentActionType.GENERATE,
            priority=4,
            estimated_tokens=3000,
            tool_hint="generate_report",
        ),
    ],
)

_EXPLORE_TEMPLATE = PlanTemplate(
    goal_type=GoalType.EXPLORE,
    description="Broad exploration of a topic or field",
    default_max_iterations=10,
    steps=[
        TemplateStep(
            description="Get a high-level overview of the topic",
            action_type=AgentActionType.SEARCH,
            priority=1,
            estimated_tokens=3000,
            tool_hint="search_wikipedia",
        ),
        TemplateStep(
            description="Identify key subtopics and areas of interest",
            action_type=AgentActionType.SEARCH,
            priority=2,
            estimated_tokens=4000,
            tool_hint="search_academic",
        ),
        TemplateStep(
            description="Find introductory and intermediate resources",
            action_type=AgentActionType.SEARCH,
            priority=3,
            estimated_tokens=4000,
            tool_hint="search_web",
        ),
        TemplateStep(
            description="Compile exploration guide with next steps",
            action_type=AgentActionType.GENERATE,
            priority=4,
            estimated_tokens=3000,
            tool_hint="generate_report",
        ),
    ],
)


class PlanTemplates:
    _TEMPLATES: dict[GoalType, PlanTemplate] = {
        GoalType.LEARN: _LEARN_TEMPLATE,
        GoalType.RESEARCH: _RESEARCH_TEMPLATE,
        GoalType.IMPLEMENT: _IMPLEMENT_TEMPLATE,
        GoalType.COMPARE: _COMPARE_TEMPLATE,
        GoalType.TROUBLESHOOT: _TROUBLESHOOT_TEMPLATE,
        GoalType.EXPLORE: _EXPLORE_TEMPLATE,
    }

    _DETECTION_KEYWORDS: dict[GoalType, tuple[str, ...]] = {
        GoalType.LEARN: (
            "learn", "understand", "study", "basics", "fundamentals",
            "tutorial", "beginner", "introduction", "how to",
        ),
        GoalType.RESEARCH: (
            "research", "investigate", "analyze", "survey",
            "literature", "papers", "deep dive", "comprehensive",
        ),
        GoalType.IMPLEMENT: (
            "implement", "build", "create", "code", "develop",
            "program", "write", "construct", "make",
        ),
        GoalType.COMPARE: (
            "compare", "versus", "vs", "difference", "better",
            "which is", "pros and cons", "advantages",
        ),
        GoalType.TROUBLESHOOT: (
            "fix", "error", "problem", "issue", "debug",
            "not working", "broken", "troubleshoot", "resolve",
        ),
        GoalType.EXPLORE: (
            "explore", "discover", "overview", "what is",
            "tell me about", "explain", "describe",
        ),
    }

    def get_template(self, goal_type: GoalType) -> PlanTemplate:
        return self._TEMPLATES.get(goal_type, _EXPLORE_TEMPLATE)

    def detect_goal_type(self, goal: str, intent: str = "") -> GoalType:
        if intent:
            try:
                return GoalType(intent.strip().lower())
            except ValueError:
                pass
        goal_lower = goal.strip().lower()
        scores: dict[GoalType, int] = {}
        for goal_type, keywords in self._DETECTION_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in goal_lower:
                    score += 1
            scores[goal_type] = score
        best_type = max(scores, key=scores.get)
        if scores[best_type] == 0:
            return GoalType.EXPLORE
        return best_type

    def get_all_templates(self) -> dict[GoalType, PlanTemplate]:
        return dict(self._TEMPLATES)

    def get_template_steps(self, goal_type: GoalType) -> list[TemplateStep]:
        template = self.get_template(goal_type)
        return list(template.steps)