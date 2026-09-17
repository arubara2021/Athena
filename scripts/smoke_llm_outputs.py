from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

TESTS = [
    ("MISTRAL", "mistral-medium-latest"),
    ("MISTRAL", "open-mistral-nemo"),
    ("MISTRAL", "mistral-small-latest"),
    ("MISTRAL", "ministral-3b-2512"),
    ("MISTRAL", "mistral-large-latest"),
    ("GROQ", "qwen/qwen3.8-27b"),
    ("GROQ", "qwen/qwen3.6-27b"),
    ("GROQ", "groq/compound-mini"),
    ("GROQ", "openai/gpt-oss-20b"),
    ("SAMBANOVA", "gemma-4-31B-it"),
    ("SAMBANOVA", "MiniMax-M2.7"),
    ("SAMBANOVA", "DeepSeek-V3.2"),
]

PROVIDERS = {
    "MISTRAL": {
        "key": "MISTRAL_API_KEY",
        "url": "https://api.mistral.ai/v1/chat/completions",
    },
    "GROQ": {
        "key": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
    },
    "SAMBANOVA": {
        "key": "SAMBANOVA_API_KEY",
        "url": "https://api.sambanova.ai/v1/chat/completions",
    },
}

PROMPT = "Return exactly this JSON and nothing else: {\"ok\": true, \"model_worked\": true}"


def load_env() -> dict[str, str]:
    values = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({k: v for k, v in os.environ.items() if k.endswith("_API_KEY")})
    return values


def extract_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    return ""


async def test_one(client: httpx.AsyncClient, env: dict[str, str], provider: str, model: str) -> dict[str, Any]:
    config = PROVIDERS[provider]
    api_key = env.get(config["key"], "")
    if not api_key:
        return {
            "provider": provider,
            "model": model,
            "ok": False,
            "status": None,
            "text": "",
            "error": "missing api key",
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON test responder."},
            {"role": "user", "content": PROMPT},
        ],
        "temperature": 0,
        "max_tokens": 64,
    }

    try:
        response = await client.post(config["url"], headers=headers, json=payload, timeout=45)
        text = response.text[:500]
        if response.status_code != 200:
            return {
                "provider": provider,
                "model": model,
                "ok": False,
                "status": response.status_code,
                "text": "",
                "error": text,
            }

        data = response.json()
        output = extract_text(data)
        return {
            "provider": provider,
            "model": model,
            "ok": bool(output),
            "status": response.status_code,
            "text": output[:300],
            "error": "" if output else "empty output",
        }
    except Exception as exc:
        return {
            "provider": provider,
            "model": model,
            "ok": False,
            "status": None,
            "text": "",
            "error": str(exc),
        }


async def main() -> None:
    env = load_env()
    async with httpx.AsyncClient() as client:
        results = []
        for provider, model in TESTS:
            result = await test_one(client, env, provider, model)
            results.append(result)
            await asyncio.sleep(0.8)

    print("\nLLM OUTPUT SMOKE TEST")
    print("=" * 80)

    for result in results:
        mark = "✅" if result["ok"] else "❌"
        print(f"\n{mark} {result['provider']} / {result['model']}")
        print(f"status: {result['status']}")
        if result["text"]:
            print(f"output: {result['text']}")
        if result["error"]:
            print(f"error: {result['error'][:300]}")

    print("\nRECOMMENDED ACTIVE MODELS")
    print("=" * 80)
    for result in results:
        if result["ok"]:
            print(f"{result['provider']} / {result['model']}")


if __name__ == "__main__":
    asyncio.run(main())