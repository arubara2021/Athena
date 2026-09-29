from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from core import constants
from core.models import CoreModel, Difficulty, LearningPath
from core.schemas import LLMMessageSchema, LLMRequestSchema, SearchQuerySchema
from llm.guardrails import validate_llm_request, validate_llm_response
from llm.parser import parse_json_object_response
from llm.provider import LLMProviderManager
from llm.router import (
    get_fast_model_references,
    get_strong_model_references,
)
from ranking.difficulty_classifier import DifficultyClassifier
from utils.logger import get_logger
from utils.text import clean_text


class StepEnhancement(CoreModel):
    step: int
    prerequisites: list[str] = Field(default_factory=list)
    suggested_projects: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)


class EnhancedLearningPath(CoreModel):
    path: LearningPath
    enhancements: list[StepEnhancement] = Field(default_factory=list)

    def enhancement_for_step(
        self, step_number: int
    ) -> StepEnhancement | None:
        for enhancement in self.enhancements:
            if enhancement.step == step_number:
                return enhancement

        return None


class PathEnhancer:
    _SEMANTIC_TITLE_TOKENS = (
        (
            "foundation",
            "basic",
            "intro",
            "prerequisite",
            "fundamental",
        ),
        ("core", "concept", "principle", "theory"),
        (
            "applied",
            "practice",
            "implementation",
            "hands-on",
            "hands on",
            "exercise",
        ),
        (
            "advanced",
            "research",
            "frontier",
            "expert",
            "cutting",
        ),
    )

    def __init__(self, use_llm: bool = True) -> None:
        self._use_llm = bool(use_llm)
        self._logger = get_logger("ranking.path_enhancer")
        self._classifier = None

        try:
            self._classifier = DifficultyClassifier()
        except Exception:
            self._classifier = None

    async def enhance(
        self,
        learning_path: LearningPath | None,
        *args: Any,
        **kwargs: Any,
    ) -> EnhancedLearningPath:
        ranked_sources = (
            kwargs.get("ranked_sources")
            or kwargs.get("sources")
            or kwargs.get("items")
        )

        query = (
            kwargs.get("query")
            or kwargs.get("query_schema")
            or kwargs.get("topic")
        )

        mode = str(kwargs.get("mode", "") or "")

        for arg in args:
            if arg is None:
                continue

            if query is None and self._looks_like_query(arg):
                query = arg
                continue

            if ranked_sources is None and isinstance(
                arg, (list, tuple, set)
            ):
                ranked_sources = list(arg)
                continue

            if query is None:
                query = arg

        if learning_path is None:
            return EnhancedLearningPath(
                path=LearningPath(
                    topic="", level="", goal="", steps=[]
                ),
                enhancements=[],
            )

        steps = getattr(learning_path, "steps", []) or []

        if not steps:
            return EnhancedLearningPath(
                path=learning_path, enhancements=[]
            )

        learning_path = self._align_path_order(
            learning_path, query
        )

        topic, level, domains = self._extract_query_fields(query)

        if not topic:
            topic = clean_text(
                getattr(learning_path, "topic", "") or ""
            )

        if not level:
            level = clean_text(
                getattr(learning_path, "level", "") or ""
            )

        resolved_mode = self._resolve_mode(mode)
        strategy = self._strategy_for_mode(resolved_mode)

        if self._use_llm and strategy != "heuristic":
            llm_result = await self._llm_enhance(
                learning_path,
                topic,
                level,
                ranked_sources,
                strategy,
            )

            if (
                llm_result is not None
                and llm_result.enhancements
            ):
                return llm_result

        return self._heuristic_enhance(
            learning_path, topic, ranked_sources
        )

    def _resolve_mode(self, mode: str) -> str:
        text = str(mode or "").strip().lower()

        if text in constants.VALID_MODES:
            return text

        return constants.DEFAULT_MODE

    def _strategy_for_mode(self, mode: str) -> str:
        if mode == constants.MODE_DEEP:
            return "strong"

        return "heuristic"

    def _looks_like_query(self, value: Any) -> bool:
        if isinstance(value, SearchQuerySchema):
            return True

        if isinstance(value, str):
            return True

        if isinstance(value, dict):
            return any(
                key in value
                for key in (
                    "topic",
                    "goal",
                    "level",
                    "query",
                    "corrected_topic",
                    "primary_concept",
                )
            )

        return hasattr(value, "topic") or hasattr(value, "goal")

    def _extract_query_fields(
        self, query: Any
    ) -> tuple[str, str, list[str]]:
        if isinstance(query, SearchQuerySchema):
            return (
                clean_text(getattr(query, "topic", "") or ""),
                clean_text(getattr(query, "level", "") or ""),
                self._clean_string_list(
                    getattr(query, "target_domains", [])
                ),
            )

        if isinstance(query, dict):
            topic = clean_text(
                query.get("topic")
                or query.get("corrected_topic")
                or query.get("primary_concept")
                or query.get("goal")
                or ""
            )

            level = clean_text(query.get("level", "") or "")

            domains = self._clean_string_list(
                query.get("target_domains", [])
            )

            return topic, level, domains

        if isinstance(query, str):
            return clean_text(query), "", []

        if query is None:
            return "", "", []

        topic = clean_text(
            getattr(query, "topic", "")
            or getattr(query, "goal", "")
            or getattr(query, "primary_concept", "")
            or ""
        )

        level = clean_text(getattr(query, "level", "") or "")
        domains = self._clean_string_list(
            getattr(query, "target_domains", [])
        )

        return topic, level, domains

    def _align_path_order(
        self,
        learning_path: LearningPath,
        query: Any,
    ) -> LearningPath:
        steps = list(getattr(learning_path, "steps", []) or [])

        if not steps:
            return learning_path

        topic, level, domains = self._extract_query_fields(query)
        updated_steps: list[Any] = []

        for step in steps:
            resources = list(
                getattr(step, "resources", []) or []
            )

            if self._classifier is not None and resources:
                sources = [
                    item.source
                    for item in resources
                    if getattr(item, "source", None) is not None
                ]

                try:
                    classified_sources = (
                        self._classifier.classify_sources(
                            sources, level=level
                        )
                    )
                except Exception:
                    classified_sources = sources

                classified_map = {
                    item.source_id: item
                    for item in classified_sources
                }

                updated_resources: list[Any] = []

                for resource in resources:
                    source = classified_map.get(
                        resource.source.source_id,
                        resource.source,
                    )

                    if source is not resource.source:
                        resource = resource.model_copy(
                            update={"source": source}
                        )

                    updated_resources.append(resource)

                resources = updated_resources

            resources.sort(
                key=lambda item: self._difficulty_order(
                    getattr(item, "source", None)
                )
            )

            updated_steps.append(
                step.model_copy(update={"resources": resources})
            )

        semantic_order = self._semantic_ranks(updated_steps)

        if semantic_order is not None:
            order_consistent = all(
                semantic_order[index] <= semantic_order[index + 1]
                for index in range(len(semantic_order) - 1)
            )

            if not order_consistent:
                sorted_pairs = sorted(
                    zip(semantic_order, updated_steps),
                    key=lambda item: item[0],
                )
                updated_steps = [step for _, step in sorted_pairs]
        else:
            scores = [
                self._step_difficulty_score(step)
                for step in updated_steps
            ]

            if self._misordered(scores):
                updated_steps.sort(
                    key=lambda step: self._step_difficulty_score(
                        step
                    )
                )

        renumbered: list[Any] = []

        for index, step in enumerate(updated_steps, start=1):
            objective = self._annotate_objective(
                getattr(step, "objective", "") or "",
                level,
                domains,
            )

            renumbered.append(
                step.model_copy(
                    update={
                        "step": index,
                        "objective": objective,
                    }
                )
            )

        return learning_path.model_copy(
            update={"steps": renumbered}
        )

    def _semantic_ranks(
        self, steps: list[Any]
    ) -> list[int] | None:
        if not steps:
            return None

        ranks: list[int] = []

        for step in steps:
            title = clean_text(
                getattr(step, "title", "") or ""
            ).lower()

            matched_rank = None

            for rank_index, keywords in enumerate(
                self._SEMANTIC_TITLE_TOKENS
            ):
                if any(
                    keyword in title for keyword in keywords
                ):
                    matched_rank = rank_index
                    break

            if matched_rank is None:
                return None

            ranks.append(matched_rank)

        return ranks

    def _step_difficulty_score(self, step: Any) -> float:
        resources = list(getattr(step, "resources", []) or [])

        if not resources:
            return 1.0

        orders = [
            self._difficulty_order(getattr(item, "source", None))
            for item in resources
        ]

        return sum(orders) / len(orders)

    def _misordered(self, scores: list[float]) -> bool:
        for index in range(len(scores) - 1):
            if scores[index] > scores[index + 1] + 0.25:
                return True

        return False

    def _difficulty_order(self, source: Any) -> int:
        if source is None:
            return 1

        difficulty = getattr(source, "difficulty", None)

        if difficulty == Difficulty.BEGINNER:
            return 0

        if difficulty == Difficulty.INTERMEDIATE:
            return 1

        if difficulty == Difficulty.ADVANCED:
            return 2

        if self._is_beginner_source(source):
            return 0

        if self._type_value(source) == "research_paper":
            return 2

        return 1

    def _is_beginner_source(self, source: Any) -> bool:
        if source is None:
            return False

        difficulty = getattr(source, "difficulty", None)

        if difficulty == Difficulty.BEGINNER:
            return True

        platform = self._enum_value(
            getattr(source, "platform", "")
        ).lower()
        source_type = self._type_value(source)

        beginner_platforms = {
            "wikipedia",
            "wikibooks",
            "wikiversity",
            "openstax",
            "libretexts",
            "mit_ocw",
            "open_library",
            "internet_archive",
        }

        beginner_source_types = {
            "documentation",
            "course",
            "book",
            "video",
        }

        if platform in beginner_platforms:
            return True

        if source_type in beginner_source_types:
            return True

        return False

    def _annotate_objective(
        self,
        objective: str,
        level: str,
        domains: list[str],
    ) -> str:
        objective = clean_text(objective)

        domain_text = (
            ", ".join(domains) if domains else "general"
        )

        annotation = (
            "Ordering: beginner -> intermediate -> advanced | "
            f"Domain: {domain_text} | "
            f"Level: {level or 'mixed'}"
        )

        if not objective:
            return annotation

        if "Ordering:" in objective:
            return objective

        return f"{objective} | {annotation}"

    def _heuristic_enhance(
        self,
        learning_path: LearningPath,
        topic: str,
        ranked_sources: list[Any] | None = None,
    ) -> EnhancedLearningPath:
        enhancements: list[StepEnhancement] = []

        for index, step in enumerate(learning_path.steps):
            prerequisites = self._heuristic_prerequisites(
                step, index, topic
            )
            projects = self._heuristic_projects(step, topic)
            concepts = self._heuristic_concepts(
                step, ranked_sources
            )

            enhancements.append(
                StepEnhancement(
                    step=step.step,
                    prerequisites=prerequisites,
                    suggested_projects=projects,
                    key_concepts=concepts,
                )
            )

        return EnhancedLearningPath(
            path=learning_path, enhancements=enhancements
        )

    def _heuristic_prerequisites(
        self,
        step: Any,
        index: int,
        topic: str,
    ) -> list[str]:
        prerequisites: list[str] = []

        if index == 0:
            prerequisites.append(
                f"Basic familiarity with {topic}"
            )
            prerequisites.append(
                "Comfort with fundamental terminology"
            )
        else:
            prerequisites.append(
                "Completion of the previous step"
            )
            prerequisites.append(
                "Understanding of the core concepts introduced "
                "so far"
            )

        step_title = clean_text(
            getattr(step, "title", "") or ""
        ).lower()

        if (
            "implementation" in step_title
            or "apply" in step_title
            or "practice" in step_title
            or "hands-on" in step_title
        ):
            prerequisites.append(
                "A suitable workspace for hands-on practice"
            )
            prerequisites.append(
                "Familiarity with the basic tools for this topic"
            )
        elif "advanced" in step_title or "research" in step_title:
            prerequisites.append(
                "Solid grasp of the fundamentals"
            )

        return prerequisites[:4]

    def _heuristic_projects(
        self, step: Any, topic: str
    ) -> list[str]:
        step_title = clean_text(
            getattr(step, "title", "") or ""
        ).lower()
        projects: list[str] = []

        if "fundamental" in step_title or "basics" in step_title:
            projects.append(
                f"Summarize the key ideas of {topic} in your "
                f"own words"
            )
            projects.append("Create a one-page concept map")
        elif (
            "implementation" in step_title
            or "apply" in step_title
            or "practice" in step_title
            or "hands-on" in step_title
        ):
            projects.append(
                f"Create a small practical example using {topic}"
            )
            projects.append(
                "Reproduce one result or idea from the resources"
            )
        elif "advanced" in step_title or "research" in step_title:
            projects.append(
                "Write a short critique of one advanced resource"
            )
            projects.append(
                f"Identify an open question related to {topic}"
            )
        else:
            projects.append(
                f"Apply what you learned about {topic} to a "
                f"mini project"
            )

        return projects[:3]

    def _heuristic_concepts(
        self,
        step: Any,
        ranked_sources: list[Any] | None = None,
    ) -> list[str]:
        concepts: list[str] = []

        objective = clean_text(
            getattr(step, "objective", "") or ""
        )

        if objective:
            concepts.append(" ".join(objective.split()[:8]))

        for resource in getattr(step, "resources", [])[:2]:
            source = getattr(resource, "source", resource)
            title = clean_text(
                getattr(source, "title", "") or ""
            )

            if title and title not in concepts:
                concepts.append(title)

        if ranked_sources:
            try:
                items = list(ranked_sources)
            except Exception:
                items = []

            for item in items[:4]:
                source = getattr(item, "source", item)
                title = clean_text(
                    getattr(source, "title", "") or ""
                )

                if title and title not in concepts:
                    concepts.append(title)

                if len(concepts) >= 4:
                    break

        return concepts[:4]

    async def _llm_enhance(
        self,
        learning_path: LearningPath,
        topic: str,
        level: str,
        ranked_sources: list[Any] | None,
        strategy: str,
    ) -> EnhancedLearningPath | None:
        model_reference = self._select_model(strategy)

        if model_reference is None:
            return None

        max_tokens = self._max_tokens_for_strategy(strategy)

        try:
            serialized_steps = []

            for index, step in enumerate(
                getattr(learning_path, "steps", []) or []
            ):
                resources = []

                for resource in (
                    getattr(step, "resources", []) or []
                ):
                    source = getattr(resource, "source", resource)
                    title = clean_text(
                        getattr(source, "title", "") or ""
                    )

                    if title:
                        resources.append(title)

                serialized_steps.append(
                    {
                        "step": getattr(step, "step", index + 1),
                        "title": getattr(step, "title", ""),
                        "objective": getattr(
                            step, "objective", ""
                        ),
                        "resources": resources,
                    }
                )

            expected_output = {
                "enhancements": [
                    {
                        "step": 1,
                        "prerequisites": ["string"],
                        "suggested_projects": ["string"],
                        "key_concepts": ["string"],
                    }
                ]
            }

            system_prompt = (
                "You are a curriculum enhancement engine. "
                "You add prerequisites, hands-on projects, and key "
                "concepts to learning steps. "
                "Keep all wording domain-neutral so it works for "
                "any topic. "
                "The steps are already in the intended teaching "
                "order from step 1 to step N. "
                "Do NOT list other steps in this path as "
                "prerequisites. "
                "Prerequisites must be external: prior knowledge, "
                "tooling, software, mathematical foundations, or "
                "skills the learner brings to the path. "
                "For step 1, prerequisites must be entry-level "
                "knowledge the learner needs before starting. "
                "For later steps, prerequisites must be external "
                "knowledge or tools, never completion of another "
                "step in this path. "
                "Return only valid JSON. Do not invent resources."
            )

            user_prompt = (
                f"Topic: {topic}\n"
                f"Level: {level}\n"
                "Learning steps:\n"
                f"{json.dumps(serialized_steps, ensure_ascii=False, indent=2)}\n"
            )

            ranked_context = self._ranked_source_context(
                ranked_sources
            )

            if ranked_context:
                user_prompt += f"{ranked_context}\n"

            user_prompt += (
                "For each step, add prerequisites, suggested "
                "projects, and key concepts.\n"
                "Expected JSON output:\n"
                f"{json.dumps(expected_output, ensure_ascii=False, indent=2)}\n"
                "Return only valid JSON."
            )

            messages = [
                LLMMessageSchema(
                    role="system", content=system_prompt
                ),
                LLMMessageSchema(
                    role="user", content=user_prompt
                ),
            ]

            async with LLMProviderManager() as manager:
                request = LLMRequestSchema(
                    provider=model_reference.provider,
                    model=model_reference.model_id,
                    messages=messages,
                    temperature=constants.DEFAULT_TEMPERATURE,
                    max_tokens=max_tokens,
                    response_format="json_object",
                )

                request = validate_llm_request(request)
                response = await manager.complete(request)

                if response is None:
                    return None

                response = validate_llm_response(response)
                payload = parse_json_object_response(
                    response.content
                )

                return self._parse_payload(
                    learning_path, payload
                )
        except Exception as exc:
            self._logger.warning(
                f"LLM path enhancement failed, using heuristic: "
                f"{exc}"
            )
            return None

    def _select_model(self, strategy: str) -> Any:
        try:
            if strategy == "fast":
                fast_refs = get_fast_model_references(limit=1)

                if fast_refs:
                    return fast_refs[0]

                strong_refs = get_strong_model_references(limit=1)
                return strong_refs[0] if strong_refs else None

            strong_refs = get_strong_model_references(limit=1)

            if strong_refs:
                return strong_refs[0]

            fast_refs = get_fast_model_references(limit=1)
            return fast_refs[0] if fast_refs else None
        except Exception:
            return None

    def _max_tokens_for_strategy(self, strategy: str) -> int:
        if strategy == "fast":
            return constants.ENHANCER_MAX_TOKENS

        return int(constants.ENHANCER_MAX_TOKENS * 2)

    def _ranked_source_context(self, ranked_sources: Any) -> str:
        if not ranked_sources:
            return ""

        try:
            items = list(ranked_sources)
        except Exception:
            return ""

        titles: list[str] = []

        for item in items[:8]:
            source = getattr(item, "source", item)
            title = clean_text(
                getattr(source, "title", "") or ""
            )

            if title and title not in titles:
                titles.append(title)

        if not titles:
            return ""

        return "Available ranked sources:" + "".join(
            f"- {title}" for title in titles
        )

    def _parse_payload(
        self,
        learning_path: LearningPath,
        payload: dict[str, Any],
    ) -> EnhancedLearningPath:
        raw_enhancements = payload.get("enhancements", [])

        if (
            not isinstance(raw_enhancements, list)
            or not raw_enhancements
        ):
            return self._heuristic_enhance(
                learning_path, learning_path.topic
            )

        step_numbers = {
            step.step for step in learning_path.steps
        }
        enhancements: list[StepEnhancement] = []

        for item in raw_enhancements:
            if not isinstance(item, dict):
                continue

            try:
                step_number = int(item.get("step", 0))
            except Exception:
                continue

            if step_number not in step_numbers:
                continue

            enhancements.append(
                StepEnhancement(
                    step=step_number,
                    prerequisites=self._clean_string_list(
                        item.get("prerequisites")
                    ),
                    suggested_projects=self._clean_string_list(
                        item.get("suggested_projects")
                    ),
                    key_concepts=self._clean_string_list(
                        item.get("key_concepts")
                    ),
                )
            )

        if not enhancements:
            return self._heuristic_enhance(
                learning_path, learning_path.topic
            )

        return EnhancedLearningPath(
            path=learning_path, enhancements=enhancements
        )

    def _clean_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []

        cleaned = [
            clean_text(item)
            for item in value
            if isinstance(item, str)
        ]
        return [item for item in cleaned if item][:6]

    def _type_value(self, source: Any) -> str:
        return self._enum_value(
            getattr(source, "source_type", "")
        ).lower()

    def _enum_value(self, value: Any) -> str:
        return str(getattr(value, "value", value) or "")