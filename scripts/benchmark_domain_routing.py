from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from core.schemas import SearchQuerySchema
    from search.orchestrator import SearchOrchestrator
except Exception as exc:
    print("IMPORT_FAILED")
    print(str(exc))
    raise SystemExit(1)

try:
    from search.source_fetcher import load_domain_routing_config
except Exception:
    load_domain_routing_config = None

USE_LLM = os.getenv("BENCHMARK_USE_LLM", "0") == "1"
QUICK = os.getenv("BENCHMARK_QUICK", "0") == "1"
CONCURRENCY = int(os.getenv("BENCHMARK_CONCURRENCY", "6"))
REQUEST_TIMEOUT = float(os.getenv("BENCHMARK_REQUEST_TIMEOUT", "30"))
CASE_TIMEOUT_SECONDS = float(os.getenv("BENCHMARK_CASE_TIMEOUT_SECONDS", "180"))
MAX_RESULTS = int(os.getenv("BENCHMARK_MAX_RESULTS", "8"))

try:
    CONFIG_REPORT = load_domain_routing_config() if callable(load_domain_routing_config) else {}
except Exception:
    CONFIG_REPORT = {}

if not isinstance(CONFIG_REPORT, dict):
    CONFIG_REPORT = {}

BEGINNER_PLATFORMS = {
    "wikipedia",
    "wikibooks",
    "wikiversity",
    "openstax",
    "libretexts",
    "mit_ocw",
    "open_library",
    "internet_archive",
}

BEGINNER_SOURCE_TYPES = {
    "course",
    "book",
    "documentation",
    "video",
}

RESEARCH_PLATFORMS = {
    "arxiv",
    "semantic_scholar",
    "openalex",
    "core",
    "crossref",
    "europe_pmc",
    "pubmed",
    "doaj",
    "zenodo",
    "datacite",
}

BEGINNER_MARKERS = (
    "tutorial",
    "introduction",
    "intro",
    "basics",
    "basic",
    "beginner",
    "fundamentals",
    "fundamental",
    "course",
    "textbook",
    "guide",
    "explained",
    "simple",
    "easy",
    "step by step",
    "for beginners",
    "from scratch",
)

ADVANCED_MARKERS = (
    "advanced",
    "theorem",
    "lemma",
    "proof",
    "proposition",
    "novel",
    "state of the art",
    "state-of-the-art",
    "benchmark",
    "optimization",
    "convergence",
    "formal",
    "rigorous",
    "we propose",
    "we introduce",
    "we present",
    "we demonstrate",
    "architecture",
    "research paper",
)

BENCHMARK_CASES = [
    {
        "name": "chemistry_beginner",
        "query": "learn organic chemistry from basics to advanced",
        "goal": "Learn organic chemistry from basics to advanced",
        "level": "beginner",
        "expected_domains": ["chemistry", "education"],
        "expect_beginner_sources": True,
        "expect_research_sources": False,
        "allow_no_sources": False,
        "expect_error": False,
    },
    {
        "name": "computer_science_beginner",
        "query": "learn mixture of experts models from basics to advanced",
        "goal": "Learn mixture of experts models from basics to advanced",
        "level": "beginner",
        "expected_domains": ["computer_science", "education"],
        "expect_beginner_sources": True,
        "expect_research_sources": False,
        "allow_no_sources": False,
        "expect_error": False,
    },
    {
        "name": "biology_beginner",
        "query": "learn photosynthesis from basics to advanced",
        "goal": "Learn photosynthesis from basics to advanced",
        "level": "beginner",
        "expected_domains": ["biology", "education"],
        "expect_beginner_sources": True,
        "expect_research_sources": False,
        "allow_no_sources": False,
        "expect_error": False,
    },
    {
        "name": "medicine_beginner",
        "query": "learn diabetes treatment from basics to advanced",
        "goal": "Learn diabetes treatment from basics to advanced",
        "level": "beginner",
        "expected_domains": ["medicine", "education"],
        "expect_beginner_sources": True,
        "expect_research_sources": False,
        "allow_no_sources": False,
        "expect_error": False,
    },
    {
        "name": "law_beginner",
        "query": "learn constitutional law from basics",
        "goal": "Learn constitutional law from basics",
        "level": "beginner",
        "expected_domains": ["law", "education"],
        "expect_beginner_sources": True,
        "expect_research_sources": False,
        "allow_no_sources": False,
        "expect_error": False,
    },
    {
        "name": "math_beginner",
        "query": "learn calculus from basics to advanced",
        "goal": "Learn calculus from basics to advanced",
        "level": "beginner",
        "expected_domains": ["mathematics", "education"],
        "expect_beginner_sources": True,
        "expect_research_sources": False,
        "allow_no_sources": False,
        "expect_error": False,
    },
    {
        "name": "physics_beginner",
        "query": "learn quantum mechanics from basics",
        "goal": "Learn quantum mechanics from basics",
        "level": "beginner",
        "expected_domains": ["physics", "education"],
        "expect_beginner_sources": True,
        "expect_research_sources": False,
        "allow_no_sources": False,
        "expect_error": False,
    },
    {
        "name": "advanced_computer_science",
        "query": "advanced mixture of experts load balancing research papers",
        "goal": "Find advanced research papers about mixture of experts load balancing",
        "level": "advanced",
        "expected_domains": ["computer_science"],
        "expect_beginner_sources": False,
        "expect_research_sources": True,
        "allow_no_sources": False,
        "expect_error": False,
    },
    {
        "name": "off_topic_nonsense",
        "query": "quantum blockchain organic constitutional law",
        "goal": "Stress test broad unrelated domains",
        "level": "mixed",
        "expected_domains": [],
        "expect_beginner_sources": False,
        "expect_research_sources": False,
        "allow_no_sources": True,
        "expect_error": False,
    },
    {
        "name": "invalid_empty_query",
        "query": "",
        "goal": "",
        "level": "",
        "expected_domains": [],
        "expect_beginner_sources": False,
        "expect_research_sources": False,
        "allow_no_sources": True,
        "expect_error": True,
    },
    {
        "name": "overlong_query",
        "query": "learn " + ("advanced " * 200),
        "goal": "Invalid overlong query",
        "level": "",
        "expected_domains": [],
        "expect_beginner_sources": False,
        "expect_research_sources": False,
        "allow_no_sources": True,
        "expect_error": True,
    },
]


def get_field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def enum_value(value):
    return str(getattr(value, "value", value) or "")


def normalize_url_key(value):
    value = str(value or "").strip().lower()
    if not value:
        return ""
    parsed = urlparse(value)
    if not parsed.netloc:
        return value
    path = parsed.path.rstrip("/")
    return f"{parsed.netloc}{path}"


def normalize_title_key(value):
    value = str(value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def get_min_domain_confidence():
    try:
        return float(CONFIG_REPORT.get("min_confidence", 0.35))
    except Exception:
        return 0.35


def get_min_domain_sources():
    try:
        return int(CONFIG_REPORT.get("min_sources_before_domain_expansion", 8))
    except Exception:
        return 8


def is_beginner_source(source):
    platform = enum_value(get_field(source, "platform", "")).lower()
    source_type = enum_value(get_field(source, "source_type", "")).lower()
    difficulty = enum_value(get_field(source, "difficulty", "")).lower()

    metadata = get_field(source, "metadata", {}) or {}
    metadata_difficulty = ""
    if isinstance(metadata, dict):
        metadata_difficulty = str(metadata.get("difficulty", "") or "").lower()

    if difficulty == "beginner" or metadata_difficulty == "beginner":
        return True

    if platform in BEGINNER_PLATFORMS:
        return True

    if source_type in BEGINNER_SOURCE_TYPES:
        return True

    title = str(get_field(source, "title", "") or "").lower()
    abstract = str(get_field(source, "abstract", "") or "").lower()
    text = f"{title} {abstract}"

    return any(marker in text for marker in BEGINNER_MARKERS)


def is_research_source(source):
    platform = enum_value(get_field(source, "platform", "")).lower()
    source_type = enum_value(get_field(source, "source_type", "")).lower()
    return source_type == "research_paper" and platform in RESEARCH_PLATFORMS


def is_advanced_source(source):
    if is_beginner_source(source):
        return False

    source_type = enum_value(get_field(source, "source_type", "")).lower()
    platform = enum_value(get_field(source, "platform", "")).lower()

    title = str(get_field(source, "title", "") or "").lower()
    abstract = str(get_field(source, "abstract", "") or "").lower()
    text = f"{title} {abstract}"

    if source_type == "research_paper" and platform in RESEARCH_PLATFORMS:
        return True

    return any(marker in text for marker in ADVANCED_MARKERS)


def count_beginner_sources(sources):
    return sum(1 for source in sources if is_beginner_source(source))


def count_research_sources(sources):
    return sum(1 for source in sources if is_research_source(source))


def invalid_source_report(sources):
    bad = []

    for source in sources:
        title = str(get_field(source, "title", "") or "").strip()
        url = str(get_field(source, "url", "") or "").strip()
        parsed = urlparse(url)

        if not title:
            bad.append("empty_title")

        if not url or not parsed.scheme or not parsed.netloc:
            bad.append("invalid_url")

    return bad


def duplicate_report(sources):
    seen_urls = set()
    seen_titles = set()
    duplicates = []

    for source in sources:
        url_key = normalize_url_key(get_field(source, "url", ""))
        title_key = normalize_title_key(get_field(source, "title", ""))

        if url_key and url_key in seen_urls:
            duplicates.append(url_key)
        elif title_key and title_key in seen_titles:
            duplicates.append(title_key)

        if url_key:
            seen_urls.add(url_key)

        if title_key:
            seen_titles.add(title_key)

    return duplicates


def evaluate_result(case, result):
    failures = []
    warnings = []

    sources = list(get_field(result, "sources", []) or [])
    total_found = int(get_field(result, "total_found", 0) or 0)
    total_valid = int(get_field(result, "total_valid", 0) or 0)

    result_errors = list(get_field(result, "errors", []) or [])
    if result_errors:
        warnings.extend(str(item) for item in result_errors)

    if case.get("allow_no_sources") and total_found == 0:
        return failures, warnings

    if total_found == 0:
        failures.append("no_sources_found")

    if total_found > 0 and total_valid == 0:
        failures.append("no_valid_sources")

    if total_found > 0 and not sources:
        failures.append("no_final_sources")

    if not hasattr(result, "target_domains") or not hasattr(result, "domain_routed"):
        failures.append("missing_domain_fields")
    else:
        target_domains = [str(item).lower() for item in list(get_field(result, "target_domains", []) or [])]
        domain_routed = bool(get_field(result, "domain_routed", False))
        domain_confidence = float(get_field(result, "domain_confidence", 0.0) or 0.0)
        domain_platforms = [str(item).lower() for item in list(get_field(result, "domain_platforms", []) or [])]
        expected_domains = [str(item).lower() for item in list(case.get("expected_domains", []) or [])]

        if expected_domains:
            if not target_domains:
                failures.append("no_domains_detected")

            if not domain_routed:
                failures.append("domain_routing_not_active")

            for expected in expected_domains:
                if expected not in target_domains:
                    failures.append(f"missing_expected_domain:{expected}")

        if domain_confidence <= 0.0:
            warnings.append("domain_confidence_zero")
        elif domain_confidence < get_min_domain_confidence():
            warnings.append("low_domain_confidence")

        if domain_routed and not domain_platforms:
            warnings.append("domain_platforms_empty")

        if domain_platforms and sources:
            unexpected = []
            for source in sources:
                platform = enum_value(get_field(source, "platform", "")).lower()
                if platform and platform not in domain_platforms:
                    unexpected.append(platform)

            if unexpected:
                warnings.append("unexpected_platforms:" + ",".join(sorted(set(unexpected))))

    if case.get("expect_beginner_sources"):
        beginner_count = count_beginner_sources(sources)

        if beginner_count == 0:
            failures.append("no_beginner_sources_for_beginner_query")
        elif sources:
            first_source = sources[0]
            if is_advanced_source(first_source) and not is_beginner_source(first_source):
                failures.append("advanced_source_first_for_beginner_query")

        if beginner_count < 2 and len(sources) >= 3:
            warnings.append("too_few_beginner_sources")

    if case.get("expect_research_sources"):
        research_count = count_research_sources(sources)
        if research_count == 0:
            warnings.append("no_research_sources_for_advanced_query")

    invalid_sources = invalid_source_report(sources)
    if invalid_sources:
        failures.append(f"invalid_sources:{len(invalid_sources)}")

    duplicates = duplicate_report(sources)
    if duplicates:
        failures.append(f"duplicate_sources:{len(duplicates)}")

    domain_routed = bool(get_field(result, "domain_routed", False)) if hasattr(result, "domain_routed") else False
    domain_expanded = bool(get_field(result, "domain_expanded", False)) if hasattr(result, "domain_expanded") else False

    if domain_routed and not domain_expanded and total_found < get_min_domain_sources():
        warnings.append("domain_expansion_not_triggered")

    return failures, warnings


async def run_case(orchestrator, case):
    started = time.perf_counter()

    record = {
        "case": case,
        "success": False,
        "failures": [],
        "warnings": [],
        "error": None,
        "latency_ms": 0.0,
        "target_domains": [],
        "target_formats": [],
        "domain_confidence": 0.0,
        "domain_routed": False,
        "domain_expanded": False,
        "domain_platforms": [],
        "total_found": 0,
        "total_valid": 0,
        "final_sources": 0,
        "platform_distribution": {},
        "top_sources": [],
    }

    try:
        if case.get("expect_error"):
            try:
                query = SearchQuerySchema(
                    topic=case.get("query", ""),
                    goal=case.get("goal", ""),
                    level=case.get("level", ""),
                    max_results=MAX_RESULTS,
                )
                result = await asyncio.wait_for(
                    orchestrator.search(query),
                    timeout=CASE_TIMEOUT_SECONDS,
                )
                record["failures"].append("expected_error_but_succeeded")
                record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
                return record
            except Exception as exc:
                record["success"] = True
                record["error"] = str(exc)
                record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
                return record

        try:
            query = SearchQuerySchema(
                topic=case.get("query", ""),
                goal=case.get("goal", ""),
                level=case.get("level", ""),
                max_results=MAX_RESULTS,
            )
        except Exception as exc:
            record["failures"].append(f"query_schema_failed:{exc}")
            record["error"] = str(exc)
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return record

        result = await asyncio.wait_for(
            orchestrator.search(query),
            timeout=CASE_TIMEOUT_SECONDS,
        )

        if result is None:
            record["failures"].append("orchestrator_returned_none")
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return record

        sources = list(get_field(result, "sources", []) or [])

        record["target_domains"] = list(get_field(result, "target_domains", []) or [])
        record["target_formats"] = list(get_field(result, "target_formats", []) or [])
        record["domain_confidence"] = float(get_field(result, "domain_confidence", 0.0) or 0.0)
        record["domain_routed"] = bool(get_field(result, "domain_routed", False))
        record["domain_expanded"] = bool(get_field(result, "domain_expanded", False))
        record["domain_platforms"] = list(get_field(result, "domain_platforms", []) or [])
        record["total_found"] = int(get_field(result, "total_found", 0) or 0)
        record["total_valid"] = int(get_field(result, "total_valid", 0) or 0)
        record["final_sources"] = len(sources)
        record["platform_distribution"] = dict(
            Counter(enum_value(get_field(source, "platform", "")) for source in sources)
        )

        for source in sources[:5]:
            record["top_sources"].append(
                {
                    "title": str(get_field(source, "title", "") or ""),
                    "platform": enum_value(get_field(source, "platform", "")),
                    "source_type": enum_value(get_field(source, "source_type", "")),
                    "url": str(get_field(source, "url", "") or ""),
                }
            )

        failures, warnings = evaluate_result(case, result)
        record["failures"] = failures
        record["warnings"] = warnings
        record["success"] = not failures

    except asyncio.TimeoutError:
        record["failures"].append("case_timeout")
        record["error"] = f"Timed out after {CASE_TIMEOUT_SECONDS} seconds"
    except Exception as exc:
        record["failures"].append(f"unexpected_exception:{type(exc).__name__}")
        record["error"] = str(exc)

    record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return record


async def run_all(orchestrator, cases):
    records = []

    for case in cases:
        record = await run_case(orchestrator, case)
        records.append(record)
        print_case(record)

    return records


def print_case(record):
    case = record.get("case", {})
    status = "PASS" if record.get("success") else "FAIL"

    print("=" * 100)
    print(f"CASE: {case.get('name', 'unknown')}")
    print(f"STATUS: {status}")
    print(f"QUERY: {case.get('query', '')}")
    print(f"EXPECTED DOMAINS: {', '.join(case.get('expected_domains', []) or []) or 'none'}")
    print(f"DETECTED DOMAINS: {', '.join(record.get('target_domains', []) or []) or 'none'}")
    print(f"DOMAIN CONFIDENCE: {record.get('domain_confidence', 0.0)}")
    print(f"DOMAIN ROUTED: {record.get('domain_routed', False)}")
    print(f"DOMAIN EXPANDED: {record.get('domain_expanded', False)}")
    print(f"DOMAIN PLATFORMS: {', '.join(record.get('domain_platforms', []) or []) or 'none'}")
    print(f"TOTAL FOUND: {record.get('total_found', 0)}")
    print(f"TOTAL VALID: {record.get('total_valid', 0)}")
    print(f"FINAL SOURCES: {record.get('final_sources', 0)}")
    print(f"LATENCY: {record.get('latency_ms', 0.0)} ms")

    if record.get("platform_distribution"):
        print("PLATFORM DISTRIBUTION:")
        for platform, count in record.get("platform_distribution", {}).items():
            print(f"  {platform}: {count}")

    if record.get("top_sources"):
        print("TOP SOURCES:")
        for source in record.get("top_sources", []):
            print(f"  [{source.get('platform', '-')}] {source.get('title', 'Untitled')}")
            print(f"    {source.get('url', '')}")

    if record.get("failures"):
        print("FAILURES:")
        for item in record.get("failures", []):
            print(f"  {item}")

    if record.get("warnings"):
        print("WARNINGS:")
        for item in record.get("warnings", []):
            print(f"  {item}")

    if record.get("error"):
        print(f"ERROR: {record.get('error')}")


def create_orchestrator():
    try:
        return SearchOrchestrator(
            use_llm_expansion=USE_LLM,
            concurrency=CONCURRENCY,
            request_timeout=REQUEST_TIMEOUT,
        )
    except TypeError:
        return SearchOrchestrator()


async def main():
    cases = BENCHMARK_CASES[:3] if QUICK else BENCHMARK_CASES

    print("=" * 100)
    print("ATHENA DOMAIN ROUTING BENCHMARK")
    print("=" * 100)
    print(f"USE_LLM: {USE_LLM}")
    print(f"QUICK_MODE: {QUICK}")
    print(f"CASES: {len(cases)}")
    print(f"MAX_RESULTS: {MAX_RESULTS}")
    print(f"CASE_TIMEOUT_SECONDS: {CASE_TIMEOUT_SECONDS}")
    print("=" * 100)

    orchestrator = create_orchestrator()

    if hasattr(orchestrator, "__aenter__"):
        async with orchestrator:
            records = await run_all(orchestrator, cases)
    else:
        try:
            records = await run_all(orchestrator, cases)
        finally:
            close_method = getattr(orchestrator, "close", None)
            if callable(close_method):
                close_result = close_method()
                if asyncio.iscoroutine(close_result):
                    await close_result

    latencies = [float(record.get("latency_ms", 0.0)) for record in records if record.get("latency_ms")]
    failed_case_names = [record.get("case", {}).get("name", "unknown") for record in records if not record.get("success")]

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(records),
        "successful_cases": sum(1 for record in records if record.get("success")),
        "failed_cases": failed_case_names,
        "warnings_count": sum(len(record.get("warnings", [])) for record in records),
        "average_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "max_latency_ms": round(max(latencies), 1) if latencies else 0.0,
        "records": records,
    }

    output_dir = ROOT / "output" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"domain_routing_benchmark_{timestamp}.json"
    output_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("=" * 100)
    print(f"BENCHMARK SAVED: {output_file}")
    print(f"TOTAL CASES: {summary['total_cases']}")
    print(f"SUCCESSFUL: {summary['successful_cases']}")
    print(f"FAILED CASES: {len(summary['failed_cases'])}")
    print(f"WARNINGS: {summary['warnings_count']}")
    print(f"AVERAGE LATENCY: {summary['average_latency_ms']} ms")
    print(f"MAX LATENCY: {summary['max_latency_ms']} ms")

    if failed_case_names:
        print("FAILED CASE NAMES:")
        for name in failed_case_names:
            print(f"  {name}")

    print("=" * 100)

    raise SystemExit(1 if failed_case_names else 0)


if __name__ == "__main__":
    asyncio.run(main())