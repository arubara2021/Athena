from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from core import constants
except Exception as exc:
    print(f"Failed to import core package from project root {ROOT}: {exc}")
    raise SystemExit(1)

import httpx

OUTPUT_FILE = ROOT / "api_check_results.json"
TIMEOUT = 15.0

_HEADERS = {
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": constants.USER_AGENT,
}

PLATFORM_CHECKS: list[dict[str, Any]] = [
    {
        "platform": "pubmed",
        "name": "PubMed E-utilities",
        "checks": [
            {
                "url": f"{constants.PUBMED_BASE_URL}/esearch.fcgi?db=pubmed&term=test&retmax=1&retmode=json",
                "kind": "json",
            }
        ],
    },
    {
        "platform": "europe_pmc",
        "name": "Europe PMC",
        "checks": [
            {
                "url": f"{constants.EUROPE_PMC_BASE_URL}/search?query=test&format=json&pageSize=1",
                "kind": "json",
            }
        ],
    },
    {
        "platform": "doaj",
        "name": "DOAJ API",
        "checks": [
            {
                "url": f"{constants.DOAJ_BASE_URL}/search/articles/test?pageSize=1",
                "kind": "json",
            }
        ],
    },
    {
        "platform": "crossref",
        "name": "Crossref API",
        "checks": [
            {
                "url": f"{constants.CROSSREF_BASE_URL}/works?query=test&rows=1",
                "kind": "json",
            }
        ],
    },
    {
        "platform": "open_library",
        "name": "Open Library",
        "checks": [
            {
                "url": f"{constants.OPEN_LIBRARY_BASE_URL}/search.json?q=test&limit=1",
                "kind": "json",
            }
        ],
    },
    {
        "platform": "internet_archive",
        "name": "Internet Archive",
        "checks": [
            {
                "url": f"{constants.INTERNET_ARCHIVE_BASE_URL}/advancedsearch.php?q=test&fl[]=identifier&rows=1&output=json",
                "kind": "json",
            }
        ],
    },
    {
        "platform": "wikibooks",
        "name": "Wikibooks",
        "checks": [
            {
                "url": "https://en.wikibooks.org/w/rest.php/v1/search/page?q=test&limit=1",
                "kind": "json",
            }
        ],
    },
    {
        "platform": "openstax",
        "name": "OpenStax",
        "checks": [
            {
                "url": "https://openstax.org/search?search_string=test",
                "kind": "html",
            }
        ],
    },
    {
        "platform": "mit_ocw",
        "name": "MIT OpenCourseWare",
        "checks": [
            {
                "url": "https://ocw.mit.edu/search/?t=test",
                "kind": "html",
            }
        ],
    },
    {
        "platform": "libretexts",
        "name": "LibreTexts",
        "checks": [
            {
                "url": "https://chem.libretexts.org/Special:Search?search=test&full=1",
                "kind": "html",
                "marker": "/wiki/",
            },
            {
                "url": "https://socialsci.libretexts.org/Special:Search?search=test&full=1",
                "kind": "html",
                "marker": "/wiki/",
            },
            {
                "url": "https://bio.libretexts.org/Special:Search?search=test&full=1",
                "kind": "html",
                "marker": "/wiki/",
            },
            {
                "url": "https://chem.libretexts.org/w/api.php?action=query&list=search&srsearch=test&srlimit=1&format=json",
                "kind": "json",
            },
        ],
    },
    {
        "platform": "wikiversity",
        "name": "Wikiversity",
        "checks": [
            {
                "url": "https://en.wikiversity.org/w/rest.php/v1/search/page?q=test&limit=1",
                "kind": "json",
            }
        ],
    },
    {
        "platform": "arxiv",
        "name": "arXiv API",
        "checks": [
            {
                "url": f"{constants.ARXIV_BASE_URL}/query?search_query=all:test&max_results=1",
                "kind": "json",
            }
        ],
    },
    {
        "platform": "openalex",
        "name": "OpenAlex",
        "checks": [
            {
                "url": f"{constants.OPENALEX_BASE_URL}/works?search=test&per_page=1",
                "kind": "json",
            }
        ],
    },
    {
        "platform": "wikipedia",
        "name": "Wikipedia",
        "checks": [
            {
                "url": "https://en.wikipedia.org/w/rest.php/v1/search/page?q=test&limit=1",
                "kind": "json",
            }
        ],
    },
    {
        "platform": "github",
        "name": "GitHub API",
        "checks": [
            {
                "url": f"{constants.GITHUB_BASE_URL}/search/repositories?q=test&per_page=1",
                "kind": "json",
            }
        ],
    },
    {
        "platform": "huggingface",
        "name": "Hugging Face",
        "checks": [
            {
                "url": f"{constants.HUGGINGFACE_BASE_URL}/models?search=test&limit=1",
                "kind": "json",
            }
        ],
    },
    {
        "platform": "semantic_scholar",
        "name": "Semantic Scholar",
        "checks": [
            {
                "url": f"{constants.SEMANTIC_SCHOLAR_BASE_URL}/paper/search?query=test&limit=1",
                "kind": "json",
            }
        ],
    },
]


async def check_route(client: httpx.AsyncClient, route: dict[str, Any]) -> dict[str, Any]:
    delay = 1.0
    last_error = None

    for attempt in range(2):
        start = time.perf_counter()

        try:
            response = await client.get(route["url"], timeout=TIMEOUT)
        except httpx.TimeoutException:
            last_error = f"Request timed out after {TIMEOUT}s"
            if attempt == 0:
                await asyncio.sleep(delay)
                continue

            return {
                "url": route["url"],
                "status": "timeout",
                "http_status": None,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "has_data": False,
                "error": last_error,
            }
        except httpx.ConnectError as exc:
            last_error = str(exc)[:200]
            if attempt == 0:
                await asyncio.sleep(delay)
                continue

            return {
                "url": route["url"],
                "status": "connection_error",
                "http_status": None,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "has_data": False,
                "error": last_error,
            }
        except Exception as exc:
            last_error = str(exc)[:200]
            if attempt == 0:
                await asyncio.sleep(delay)
                continue

            return {
                "url": route["url"],
                "status": "error",
                "http_status": None,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "has_data": False,
                "error": last_error,
            }

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        status_code = response.status_code

        if status_code >= 500 and attempt == 0:
            await asyncio.sleep(delay)
            continue

        if status_code == 200:
            has_data = False
            kind = str(route.get("kind", "json")).lower()

            if kind == "json":
                try:
                    data = response.json()
                    if isinstance(data, dict) and len(data) > 0:
                        has_data = True
                    elif isinstance(data, list) and len(data) > 0:
                        has_data = True
                except Exception:
                    has_data = False
            else:
                text = response.text
                marker = str(route.get("marker", "") or "")

                if marker:
                    has_data = len(text) > 50 and marker in text
                else:
                    has_data = len(text) > 50

            return {
                "url": route["url"],
                "status": "ok",
                "http_status": status_code,
                "latency_ms": latency_ms,
                "has_data": has_data,
                "error": None,
            }

        if status_code == 429:
            status = "rate_limited"
        elif status_code in (401, 403):
            status = "auth_required"
        elif status_code == 404:
            status = "not_found"
        elif status_code >= 500:
            status = "server_error"
        else:
            status = f"http_{status_code}"

        return {
            "url": route["url"],
            "status": status,
            "http_status": status_code,
            "latency_ms": latency_ms,
            "has_data": False,
            "error": None,
        }

    return {
        "url": route["url"],
        "status": "error",
        "http_status": None,
        "latency_ms": 0.0,
        "has_data": False,
        "error": last_error,
    }


async def check_platform(client: httpx.AsyncClient, check: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    last: dict[str, Any] | None = None

    for route in check["checks"]:
        route_result = await check_route(client, route)
        last = route_result

        if route_result["status"] == "ok":
            best = route_result
            if route_result["has_data"]:
                break

    if best is None:
        best = last

    if best is None:
        best = {
            "url": "",
            "status": "error",
            "http_status": None,
            "latency_ms": 0.0,
            "has_data": False,
            "error": "No route checked",
        }

    return {
        "platform": check["platform"],
        "name": check["name"],
        "url": best["url"],
        "auth_required": False,
        "status": best["status"],
        "http_status": best["http_status"],
        "latency_ms": best["latency_ms"],
        "error": best.get("error"),
        "has_data": best["has_data"],
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


async def run_all_checks() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(follow_redirects=True, headers=_HEADERS) as client:
        for check in PLATFORM_CHECKS:
            result = await check_platform(client, check)
            results.append(result)

            status_icon = "OK" if result["status"] == "ok" else "FAIL"

            print(
                f"  [{status_icon}] {result['name']:<25} "
                f"status={result['status']:<15} "
                f"latency={result['latency_ms']:.0f}ms "
                f"data={'yes' if result['has_data'] else 'no'}"
            )

    return results


def build_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_count = sum(1 for r in results if r["status"] == "ok")
    fail_count = sum(1 for r in results if r["status"] != "ok")
    with_data = sum(1 for r in results if r["has_data"])

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "total_platforms": len(results),
        "ok_count": ok_count,
        "fail_count": fail_count,
        "platforms_with_data": with_data,
        "platforms": results,
    }


async def main() -> None:
    print("=" * 60)
    print("ATHENA API HEALTH CHECK")
    print("=" * 60)
    print()

    results = await run_all_checks()
    report = build_report(results)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"OK: {report['ok_count']} / {report['total_platforms']}")
    print(f"Platforms with data: {report['platforms_with_data']}")
    print()


if __name__ == "__main__":
    asyncio.run(main())