import asyncio
import json
import os
from typing import Any

import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

PROVIDERS = [
    {
        "name": "SAMBANOVA",
        "env": "SAMBANOVA_API_KEY",
        "url": "https://api.sambanova.ai/v1/models",
        "auth": "bearer"
    },
    {
        "name": "GROQ",
        "env": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/models",
        "auth": "bearer"
    },
    {
        "name": "GOOGLE",
        "env": "GOOGLE_API_KEY",
        "url": "https://generativelanguage.googleapis.com/v1beta/models",
        "auth": "google"
    },
    {
        "name": "MISTRAL",
        "env": "MISTRAL_API_KEY",
        "url": "https://api.mistral.ai/v1/models",
        "auth": "bearer"
    },
    {
        "name": "NVIDIA",
        "env": "NVIDIA_API_KEY",
        "url": "https://integrate.api.nvidia.com/v1/models",
        "auth": "bearer"
    }
]


def mask_key(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if len(value) <= 4:
        return "****"
    return f"****{value[-4:]}"


def extract_models(payload: Any) -> list[str]:
    items: list[Any] = []

    if isinstance(payload, dict):
        for key in ("data", "models", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                items = value
                break

        if not items and payload.get("object") == "list":
            data = payload.get("data")
            if isinstance(data, list):
                items = data

    elif isinstance(payload, list):
        items = payload

    models = set()

    for item in items:
        if isinstance(item, str):
            models.add(item)
        elif isinstance(item, dict):
            for field in ("id", "name", "model", "slug", "displayName"):
                candidate = item.get(field)
                if isinstance(candidate, str) and candidate:
                    models.add(candidate)
                    break

    return sorted(models)


async def request_once(
    client: httpx.AsyncClient,
    provider: dict[str, str],
    key: str
) -> tuple[bool, list[str], str]:
    headers = {
        "Accept": "application/json"
    }

    if provider["auth"] == "bearer":
        headers["Authorization"] = f"Bearer {key}"
    elif provider["auth"] == "google":
        headers["x-goog-api-key"] = key

    try:
        response = await client.get(provider["url"], headers=headers)

        if response.status_code == 200:
            try:
                payload = response.json()
            except Exception:
                return False, [], "invalid_json_response"

            return True, extract_models(payload), "ok"

        return False, [], f"http_{response.status_code}"

    except httpx.TimeoutException:
        return False, [], "timeout"

    except httpx.HTTPError as exc:
        return False, [], exc.__class__.__name__

    except Exception as exc:
        return False, [], exc.__class__.__name__


async def check_provider(
    client: httpx.AsyncClient,
    provider: dict[str, str]
) -> dict[str, Any]:
    key = os.getenv(provider["env"], "").strip()

    result: dict[str, Any] = {
        "provider": provider["name"],
        "env_var": provider["env"],
        "key_present": bool(key),
        "key_masked": mask_key(key),
        "working": False,
        "model_count": 0,
        "models": [],
        "error": None
    }

    if not key:
        result["error"] = "missing_api_key"
        return result

    last_error = "unknown_error"

    for attempt in range(3):
        if attempt > 0:
            await asyncio.sleep(0.5 * attempt)

        working, models, error = await request_once(client, provider, key)

        if working:
            result["working"] = True
            result["models"] = models
            result["model_count"] = len(models)
            result["error"] = None
            return result

        last_error = error

    result["error"] = last_error
    return result


async def main() -> None:
    timeout = httpx.Timeout(30.0, connect=10.0)
    limits = httpx.Limits(max_connections=5, max_keepalive_connections=2)

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        follow_redirects=True
    ) as client:
        tasks = [check_provider(client, provider) for provider in PROVIDERS]
        results = await asyncio.gather(*tasks)

    working_count = sum(1 for item in results if item["working"])

    summary = {
        "total_providers": len(results),
        "working_providers": working_count,
        "not_working_providers": len(results) - working_count,
        "results": results
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())