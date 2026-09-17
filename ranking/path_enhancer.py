from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from core import constants
from llm.guardrails import validate_llm_request, validate_llm_response
from llm.parser import parse_json_object_response
from llm.provider import LLMProviderManager
from llm.router import get_strong_model_references
from core.models import CoreModel, LearningPath
from core.schemas import LLMMessageSchema, LLMRequestSchema, SearchQuerySchema
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

    def enhancement_for_step(self, step_number: int) -> StepEnhancement | None:
        for enhancement in self.enhancements:
            if enhancement.step == step_number:
                return enhancement
        return None


class PathEnhancer:
    def __init__(self, use_llm: bool = True) -> None:
        self._use_llm = use_llm
        self._logger = get_logger("ranking.path_enhancer")

    async def enhance(
        self,
        learning_path: LearningPath,
        query: SearchQuerySchema | str | dict[str, Any] | None = None,
    ) -> EnhancedLearningPath:
        if learning_path is None or not learning_path.steps:
            return EnhancedLearningPath(path=learning_path, enhancements=[])
        topic = learning_path.topic
        level = learning_path.level
        if isinstance(query, SearchQuerySchema):
            topic = query.topic or topic
            level = query.level or level
        if self._use_llm:
            llm_result = await self._llm_enhance(learning_path, topic, level)
            if llm_result is not None and llm_result.enhancements:
                return llm_result
        return self._heuristic_enhance(learning_path, topic)

    def _heuristic_enhance(
        self,
        learning_path: LearningPath,
        topic: str,
    ) -> EnhancedLearningPath:
        enhancements: list[StepEnhancement] = []
        for index, step in enumerate(learning_path.steps):
            prerequisites = self._heuristic_prerequisites(step, index, topic)
            projects = self._heuristic_projects(step, topic)
            concepts = self._heuristic_concepts(step)
            enhancements.append(
                StepEnhancement(
                    step=step.step,
                    prerequisites=prerequisites,
                    suggested_projects=projects,
                    key_concepts=concepts,
                )
            )
        return EnhancedLearningPath(path=learning_path, enhancements=enhancements)

    def _heuristic_prerequisites(
        self,
        step: Any,
        index: int,
        topic: str,
    ) -> list[str]:
        prerequisites: list[str] = []
        if index == 0:
            prerequisites.append(f"Basic familiarity with {topic}")
            prerequisites.append("Comfort with fundamental terminology")
        else:
            prerequisites.append("Completion of the previous step")
            prerequisites.append("Understanding of the core concepts introduced so far")
        step_title = clean_text(step.title).lower()
        if (
            "implementation" in step_title
            or "apply" in step_title
            or "practice" in step_title
            or "hands-on" in step_title
        ):
            prerequisites.append("A suitable workspace for hands-on practice")
            prerequisites.append("Familiarity with the basic tools for this topic")
        elif "advanced" in step_title or "research" in step_title:
            prerequisites.append("Solid grasp of the fundamentals")
        return prerequisites[:4]

    def _heuristic_projects(self, step: Any, topic: str) -> list[str]:
        step_title = clean_text(step.title).lower()
        projects: list[str] = []
        if "fundamental" in step_title or "basics" in step_title:
            projects.append(f"Summarize the key ideas of {topic} in your own words")
            projects.append("Create a one-page concept map")
        elif (
            "implementation" in step_title
            or "apply" in step_title
            or "practice" in step_title
            or "hands-on" in step_title
        ):
            projects.append(f"Create a small practical example using {topic}")
            projects.append("Reproduce one result or idea from the resources")
        elif "advanced" in step_title or "research" in step_title:
            projects.append("Write a short critique of one advanced resource")
            projects.append(f"Identify an open question related to {topic}")
        else:
            projects.append(f"Apply what you learned about {topic} to a mini project")
        return projects[:3]

    def _heuristic_concepts(self, step: Any) -> list[str]:
        concepts: list[str] = []
        objective = clean_text(step.objective or "")
        if objective:
            words = objective.split()[:8]
            concepts.append(" ".join(words))
        for resource in step.resources[:2]:
            concepts.append(resource.source.title)
        return concepts[:4]

    async def _llm_enhance(
        self,
        learning_path: LearningPath,
        topic: str,
        level: str,
    ) -> EnhancedLearningPath | None:
        try:
            model_references = get_strong_model_references(limit=3)
            if not model_references:
                return None
            serialized_steps = [
                {
                    "step": step.step,
                    "title": step.title,
                    "objective": step.objective,
                    "resources": [resource.source.title for resource in step.resources],
                }
                for step in learning_path.steps
            ]
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
                "You add prerequisites, hands-on projects, and key concepts to learning steps. "
                "Keep all wording domain-neutral so it works for any topic. "
                "Return only valid JSON. Do not invent resources."
            )
            user_prompt = (
                f"Topic: {topic}\n"
                f"Level: {level}\n"
                f"Learning steps:\n{json.dumps(serialized_steps, ensure_ascii=False, indent=2)}\n"
                "For each step, add prerequisites, suggested projects, and key concepts.\n"
                f"Expected JSON output:\n{json.dumps(expected_output, ensure_ascii=False, indent=2)}\n"
                "Return only valid JSON."
            )
            messages = [
                LLMMessageSchema(role="system", content=system_prompt),
                LLMMessageSchema(role="user", content=user_prompt),
            ]
            async with LLMProviderManager() as manager:
                for model_reference in model_references:
                    try:
                        request = LLMRequestSchema(
                            provider=model_reference.provider,
                            model=model_reference.model_id,
                            messages=messages,
                            temperature=constants.DEFAULT_TEMPERATURE,
                            max_tokens=2000,
                            response_format="json_object",
                        )
                        request = validate_llm_request(request)
                        response = await manager.complete(request)
                        if response is None:
                            continue
                        response = validate_llm_response(response)
                        payload = parse_json_object_response(response.content)
                        return self._parse_payload(learning_path, payload)
                    except Exception:
                        continue
            return None
        except Exception as exc:
            self._logger.warning(f"LLM path enhancement failed, using heuristic: {exc}")
            return None

    def _parse_payload(
        self,
        learning_path: LearningPath,
        payload: dict[str, Any],
    ) -> EnhancedLearningPath:
        raw_enhancements = payload.get("enhancements", [])
        if not isinstance(raw_enhancements, list) or not raw_enhancements:
            return self._heuristic_enhance(learning_path, learning_path.topic)
        step_numbers = {step.step for step in learning_path.steps}
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
                    prerequisites=self._clean_string_list(item.get("prerequisites")),
                    suggested_projects=self._clean_string_list(item.get("suggested_projects")),
                    key_concepts=self._clean_string_list(item.get("key_concepts")),
                )
            )
        if not enhancements:
            return self._heuristic_enhance(learning_path, learning_path.topic)
        return EnhancedLearningPath(path=learning_path, enhancements=enhancements)

    def _clean_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned = [clean_text(item) for item in value if isinstance(item, str)]
        return [item for item in cleaned if item][:6]