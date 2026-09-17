from __future__ import annotations

import json
import re
from typing import Any

from core import constants
from utils.text import clean_text, truncate_text


def _to_serializable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if isinstance(value, list):
        return [_to_serializable(item) for item in value]

    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}

    return value


def _dump(value: Any) -> str:
    return json.dumps(
        _to_serializable(value),
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _clean(value: Any, max_length: int = 2000) -> str:
    return truncate_text(clean_text(value), max_length=max_length, suffix="")


def strict_json_rules() -> str:
    return (
        "Return only valid JSON. "
        "Do not return markdown. "
        "Do not return explanations outside JSON. "
        "Do not invent sources, links, papers, or facts."
    )


def _detect_level_intent(goal: str, level: str) -> str:
    goal_text = clean_text(goal).lower()
    level_text = clean_text(level).lower()

    beginner_signals = (
        "beginner", "basics", "basic", "fundamental", "intro",
        "from scratch", "no prior", "starter", "new to",
    )

    advanced_signals = (
        "advanced", "expert", "in depth", "in-depth", "deep dive",
        "research", "state of the art", "cutting edge",
    )

    is_beginner = any(
        signal in goal_text or signal in level_text
        for signal in beginner_signals
    )

    is_advanced = any(
        signal in goal_text or signal in level_text
        for signal in advanced_signals
    )

    if is_beginner and not is_advanced:
        return "beginner"

    if is_advanced and not is_beginner:
        return "advanced"

    return "mixed"


def _extract_primary_concept(topic: str) -> str:
    cleaned = clean_text(topic).lower()
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)

    separators = [" from ", " in ", " for ", " with ", " using ", " via ", " of "]

    for sep in separators:
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0].strip()
            break

    return cleaned.strip()


def _level_rules(goal: str, level: str) -> str:
    intent = _detect_level_intent(goal, level)

    if intent == "beginner":
        return (
            "MANDATORY LEVEL RULES (beginner):\n"
            "- The Level and Goal override citations, authority, and recency.\n"
            "- Each source includes a difficulty field. Use it.\n"
            "- Sources with difficulty advanced MUST receive score <= 0.30 unless clearly introductory.\n"
            "- Sources that teach the topic from fundamentals MUST receive score >= 0.60.\n"
            "- Prefer in this order: documentation, tutorials, courses, videos, beginner blogs, repositories, then papers.\n"
            "- A famous or highly-cited advanced paper is NOT a good beginner source.\n"
            "- Do not rank off-topic advanced research above on-topic beginner content.\n"
        )

    if intent == "advanced":
        return (
            "MANDATORY LEVEL RULES (advanced):\n"
            "- Prefer depth, novelty, and technical rigor.\n"
            "- Prefer in this order: research papers, advanced repositories, technical documentation, models.\n"
            "- Sources with difficulty beginner MUST receive score <= 0.40.\n"
            "- Prioritize sources that push beyond fundamentals.\n"
        )

    return (
        "MANDATORY LEVEL RULES (mixed):\n"
        "- Balance fundamentals with depth.\n"
        "- Prefer sources that cover the topic from basics through to advanced.\n"
    )


def _topic_relevance_rules(topic: str) -> str:
    primary = _extract_primary_concept(topic)

    return (
        "MANDATORY TOPIC RELEVANCE RULES:\n"
        f"- The PRIMARY CONCEPT the user wants to learn is: \"{primary}\"\n"
        "- A source MUST be primarily about this specific concept to score above 0.50.\n"
        "- A source that only mentions the concept in passing (e.g., as one of many metrics, tools, or references) MUST score <= 0.20.\n"
        "- Matching the general field or domain alone is NOT sufficient. The specific concept must be central to the source.\n"
        "- If the topic is \"X from Y\" or \"X in Y\", the source must be about X specifically, not just about the field Y.\n"
        "- A source about the field Y that does not discuss X must score <= 0.10.\n"
        "- When in doubt about whether a source is truly about the primary concept, score it lower.\n"
    )


def _learning_path_level_rules(goal: str, level: str) -> str:
    intent = _detect_level_intent(goal, level)

    if intent == "beginner":
        return (
            "MANDATORY LEVEL RULES (beginner learning path):\n"
            "- Step 1 MUST use beginner-level sources only (difficulty=beginner, or tutorials, documentation, courses, videos, Wikipedia).\n"
            "- Do NOT assign an advanced research paper as the primary resource for Step 1.\n"
            "- An advanced paper may only appear in later steps after fundamentals are covered.\n"
            "- If beginner sources are available, they MUST be used before intermediate or advanced sources.\n"
            "- Each step's resources must match the step's position: early steps = easier sources, later steps = harder sources.\n"
            "- The sources are already ordered by level-fit. Prefer sources that appear earlier in the list for early steps.\n"
        )

    if intent == "advanced":
        return (
            "MANDATORY LEVEL RULES (advanced learning path):\n"
            "- Prioritize research papers and technically deep sources.\n"
            "- Beginner-only sources should only appear if they provide necessary context.\n"
            "- Order steps from foundational theory to cutting-edge research.\n"
        )

    return (
        "MANDATORY LEVEL RULES (mixed learning path):\n"
        "- Balance fundamentals with depth.\n"
        "- Early steps should use simpler sources, later steps can use advanced sources.\n"
    )


def _resource_step_alignment_rules() -> str:
    return (
        "MANDATORY RESOURCE-STEP ALIGNMENT RULES:\n"
        "- Each resource assigned to a step MUST be directly relevant to that step's title and objective.\n"
        "- Do NOT assign a resource about topic X to a step about a completely different topic Y.\n"
        "- The resource's title or abstract must relate to the step's title or objective.\n"
        "- If no available source matches a step's topic, do NOT create that step.\n"
        "- Every step must have at least one resource that is genuinely about the step's subject.\n"
        "- Do NOT force-fit unrelated sources into steps just to fill them.\n"
    )


def system_research_ranker() -> str:
    return (
        "You are an autonomous AI research ranking engine. "
        "Rank only sources that directly match the user's topic and intent. "
        "Reject sources that are unrelated, loosely related, or only incidentally mention the topic. "
        "You MUST honor the requested level and goal above all other signals. "
        "You MUST prioritize sources that are specifically about the primary concept, not just the general field. "
        "For beginner or basics requests, prefer tutorials, documentation, courses, and introductory sources over advanced research papers. "
        "For advanced requests, prefer research papers and technically deep sources. "
        "Prefer official documentation, maintained repositories, and high-quality courses when they fit the level. "
        f"{strict_json_rules()}"
    )


def system_learning_path_builder() -> str:
    return (
        "You are an autonomous AI curriculum architect. "
        "You create structured learning paths that exactly match the user's topic, goal, and level. "
        "Each step must have a clear objective and must reference only provided source IDs. "
        "Do not invent new resources. "
        "Do not include irrelevant resources. "
        "You MUST honor the requested level above all other signals. "
        "Every resource you assign to a step MUST be directly relevant to that step's title and objective. "
        "For beginner requests, Step 1 must use beginner sources such as tutorials, documentation, courses, videos, or Wikipedia. "
        "Never assign an advanced research paper as the primary resource of Step 1 for a beginner request. "
        f"{strict_json_rules()}"
    )


def system_consensus_judge() -> str:
    return (
        "You are a consensus judge for a multi-model autonomous AI system. "
        "You compare multiple model outputs and choose the most accurate, complete, and safe answer. "
        "If outputs disagree, resolve the disagreement using logic and evidence present in the outputs only. "
        "You must always return the final answer under the key named exactly final_output. "
        f"{strict_json_rules()}"
    )


def build_source_ranking_prompt(
    topic: str,
    goal: str,
    level: str,
    sources: Any,
) -> str:
    expected_output = {
        "rankings": [
            {
                "source_id": "string",
                "rank": 1,
                "score": 0.98,
                "confidence": 0.9,
                "difficulty": "beginner | intermediate | advanced",
                "reason": "string",
            }
        ]
    }

    return (
        f"Topic: {_clean(topic)}\n"
        f"Goal: {_clean(goal)}\n"
        f"Level: {_clean(level)}\n"
        "Rank the following sources from best to worst for this exact learning goal.\n"
        "Use only the provided sources.\n"
        "Reject any source that does not directly relate to the topic.\n"
        f"{_topic_relevance_rules(topic)}\n"
        f"{_level_rules(goal, level)}\n"
        "Do not reward citations, authority, or recency when the source is off-topic or mismatched to the level.\n"
        "Score must be between 0 and 1."
    "SCORING CALIBRATION:\n"
    "- A source that is DIRECTLY about the primary concept and matches the level: score 0.80-1.00\n"
    "- A source that is RELEVANT but not perfectly matched: score 0.50-0.79\n"  
    "- A source that is TANGENTIALLY related: score 0.20-0.49\n"
    "- A source that is OFF-TOPIC or wrong level: score 0.01-0.19\n"
    "- Do NOT cluster all scores near zero. Use the full range.\n"
        "Confidence must be between 0 and 1.\n"
        f"Sources:\n{_dump(sources)}\n"
        f"Expected JSON output:\n{_dump(expected_output)}\n"
        f"{strict_json_rules()}"
    )


def build_learning_path_prompt(
    topic: str,
    goal: str,
    level: str,
    ranked_sources: Any,
) -> str:
    expected_output = {
        "topic": "string",
        "level": "string",
        "goal": "string",
        "steps": [
            {
                "step": 1,
                "title": "string",
                "objective": "string",
                "estimated_minutes": 60,
                "resource_source_ids": ["string"],
            }
        ],
    }

    return (
        f"Topic: {_clean(topic)}\n"
        f"Goal: {_clean(goal)}\n"
        f"Level: {_clean(level)}\n"
        "Build a learning path using only the provided ranked sources.\n"
        "Use only source IDs from the ranked sources.\n"
        "Do not include irrelevant sources.\n"
        f"{_topic_relevance_rules(topic)}\n"
        f"{_learning_path_level_rules(goal, level)}\n"
        f"{_resource_step_alignment_rules()}\n"
        "Order steps logically from fundamentals to advanced topics.\n"
        "IMPORTANT: You MUST generate between 3 and 6 steps. Never generate fewer than 3 steps.\n"
        "Each step must use at least one source from the ranked sources list.\n"
        f"Ranked sources:\n{_dump(ranked_sources)}\n"
        f"Expected JSON output:\n{_dump(expected_output)}\n"
        f"{strict_json_rules()}"
    )


def build_consensus_judge_prompt(
    task_name: str,
    original_prompt: str,
    votes: Any,
) -> str:
    required_schema = {
        "final_output": {},
        "agreement_score": 0.85,
        "reasoning": "string",
    }

    lines = [
        f"Task: {_clean(task_name)}\n",
        f"Original prompt:\n{_clean(original_prompt, max_length=6000)}\n",
        f"Model outputs:\n{_dump(votes)}\n",
        "Choose the single best final output from the model outputs.\n",
        "\n",
        "MANDATORY RESPONSE FORMAT:\n",
        "Return ONE valid JSON object with EXACTLY these three keys:\n",
        "1. \"final_output\" - the complete selected output, NOT a summary and NOT a description.\n",
        "2. \"agreement_score\" - a number between 0 and 1.\n",
        "3. \"reasoning\" - a short string explaining your choice.\n",
        "\n",
        "CRITICAL RULES:\n",
        "- The key name MUST be exactly \"final_output\" (lowercase, with underscore).\n",
        "- Do NOT rename it to \"output\", \"answer\", \"result\", \"best_output\", or \"selected_output\".\n",
        "- final_output MUST contain the FULL selected output exactly as it should be returned.\n",
        "- Do NOT wrap final_output inside another object or array.\n",
        "- Do NOT return any keys other than final_output, agreement_score, and reasoning.\n",
        "- The \"final_output\" key MUST be present. A response without it is invalid.\n",
        "\n",
    ]

    normalized_task = _clean(task_name).lower()

    if "rank" in normalized_task:
        ranking_example = {
            "final_output": {
                "rankings": [
                    {
                        "source_id": "example_source_id",
                        "rank": 1,
                        "score": 0.95,
                        "confidence": 0.9,
                        "difficulty": "beginner",
                        "reason": "Directly matches the topic and the requested level.",
                    }
                ]
            },
            "agreement_score": 0.85,
            "reasoning": "Selected the ranking that best matches the user's level and goal.",
        }

        lines.append(f"CONCRETE EXAMPLE (ranking task):\n{_dump(ranking_example)}\n")

    lines.append(f"Required JSON structure:\n{_dump(required_schema)}\n")
    lines.append(strict_json_rules())

    return "".join(lines)


def build_source_summary_prompt(source: Any) -> str:
    expected_output = {
        "summary": "string",
        "difficulty": "beginner | intermediate | advanced",
        "key_topics": ["string"],
        "why_useful": "string",
    }

    return (
        "Summarize the following source for a learner.\n"
        "Do not invent facts.\n"
        "Use only the information provided.\n"
        f"Source:\n{_dump(source)}\n"
        f"Expected JSON output:\n{_dump(expected_output)}\n"
        f"{strict_json_rules()}"
    )


def build_difficulty_classification_prompt(source: Any) -> str:
    expected_output = {
        "difficulty": "beginner | intermediate | advanced",
        "confidence": 0.9,
        "reason": "string",
    }

    return (
        "Classify the difficulty of the following learning source.\n"
        "Use only the provided information.\n"
        f"Source:\n{_dump(source)}\n"
        f"Expected JSON output:\n{_dump(expected_output)}\n"
        f"{strict_json_rules()}"
    )


def system_universal_query_doctor() -> str:
    return (
        "You are a universal query understanding engine. "
        "Convert any raw user query into clean searchable concepts. "
        "Correct spelling mistakes. "
        "Remove conversational filler. "
        "Detect intent and learner level. "
        "Preserve acronyms and technical terms. "
        "Do not invent facts. "
        f"{strict_json_rules()}"
    )


def build_universal_query_prompt(
    topic: str,
    goal: str,
    level: str,
) -> str:
    expected_output = {
        "corrected_topic": "short corrected topic",
        "primary_concept": "short primary concept",
        "keywords": ["keyword one", "keyword two"],
        "intent": "learn | research | implement | compare | troubleshoot | explore",
        "level": "beginner | intermediate | advanced | mixed",
        "short_search_queries": ["short query one", "short query two"],
    }

    return (
        f"Raw user query: {_clean(topic)}\n"
        f"User goal: {_clean(goal)}\n"
        f"User level: {_clean(level)}\n"
        "Rules:\n"
        "- corrected_topic must be a short corrected topic of 2 to 6 keywords.\n"
        "- primary_concept must be the main concept of 2 to 5 keywords.\n"
        "- keywords must contain 3 to 8 short keywords.\n"
        "- intent must be one allowed value.\n"
        "- level must be beginner, intermediate, advanced, or mixed.\n"
        "- short_search_queries must contain 2 to 6 short keyword queries.\n"
        "- Every short_search_queries item must be 2 to 6 words.\n"
        "- Do not return conversational sentences.\n"
        "- Do not return extra keys.\n"
        "- Correct spelling mistakes.\n"
        "- Preserve acronyms and technical terms.\n"
        f"Expected JSON output:\n{_dump(expected_output)}\n"
        f"{strict_json_rules()}"
    )