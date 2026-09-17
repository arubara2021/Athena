"""
API Key & Model Discovery Script
================================
Tests all configured LLM providers, lists available models,
and verifies each one works with a test completion request.

Usage:
    python scripts/check_models.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:
    print("ERROR: httpx is not installed. Run: pip install httpx")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# Load API Keys
# ─────────────────────────────────────────────────────────────
def load_env() -> dict[str, str]:
    """Load .env file from project root."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    env: dict[str, str] = {}
    if not env_path.exists():
        print(f"WARNING: .env file not found at {env_path}")
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


ENV = load_env()

PROVIDERS = {
    "GOOGLE": {
        "key_name": "GOOGLE_API_KEY",
        "models_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "chat_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "auth_type": "query_param",  # ?key=
        "preferred_models": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
    },
    "GROQ": {
        "key_name": "GROQ_API_KEY",
        "models_url": "https://api.groq.com/openai/v1/models",
        "chat_url": "https://api.groq.com/openai/v1/chat/completions",
        "auth_type": "bearer",
        "preferred_models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-3.2-90b-vision-preview",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "qwen/qwen3-32b",
            "moonshotai/kimi-k2-instruct",
        ],
    },
    "SAMBANOVA": {
        "key_name": "SAMBANOVA_API_KEY",
        "models_url": "https://api.sambanova.ai/v1/models",
        "chat_url": "https://api.sambanova.ai/v1/chat/completions",
        "auth_type": "bearer",
        "preferred_models": [
            "Meta-Llama-3.1-405B-Instruct",
            "Meta-Llama-3.1-70B-Instruct",
            "Meta-Llama-3.1-8B-Instruct",
            "DeepSeek-V3-0324",
            "DeepSeek-V3.2",
            "Qwen2.5-72B-Instruct",
            "Qwen3-235B-A22B",
        ],
    },
    "MISTRAL": {
        "key_name": "MISTRAL_API_KEY",
        "models_url": "https://api.mistral.ai/v1/models",
        "chat_url": "https://api.mistral.ai/v1/chat/completions",
        "auth_type": "bearer",
        "preferred_models": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "open-mistral-nemo",
            "codestral-latest",
            "open-mixtral-8x22b",
        ],
    },
    "NVIDIA": {
        "key_name": "NVIDIA_API_KEY",
        "models_url": "https://integrate.api.nvidia.com/v1/models",
        "chat_url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "auth_type": "bearer",
        "preferred_models": [
            "meta/llama-3.3-70b-instruct",
            "meta/llama-3.1-70b-instruct",
            "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "nvidia/llama-3.1-nemotron-ultra-253b-v1",
            "deepseek-ai/deepseek-r1",
            "qwen/qwen3-235b-a22b-thinking",
        ],
    },
    "OPENROUTER": {
        "key_name": "OPENROUTER_API_KEY",
        "models_url": "https://openrouter.ai/api/v1/models",
        "chat_url": "https://openrouter.ai/api/v1/chat/completions",
        "auth_type": "bearer",
        "preferred_models": [
            "google/gemini-2.5-flash",
            "google/gemini-2.5-pro",
            "anthropic/claude-3.5-sonnet",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-chat",
            "qwen/qwen-3-235b-a22b",
        ],
    },
}

TEST_PROMPT = "Reply with exactly: OK"
TIMEOUT = httpx.Timeout(30.0, connect=10.0)


# ─────────────────────────────────────────────────────────────
# Provider Testers
# ─────────────────────────────────────────────────────────────
async def test_google(client: httpx.AsyncClient, api_key: str) -> dict[str, Any]:
    """Google Gemini has its own API format."""
    result = {"provider": "GOOGLE", "models": [], "working_models": [], "errors": []}

    # List models
    try:
        resp = await client.get(
            PROVIDERS["GOOGLE"]["models_url"],
            params={"key": api_key},
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            all_models = [
                m["name"].replace("models/", "")
                for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            result["models"] = all_models
        else:
            result["errors"].append(f"List failed: HTTP {resp.status_code} - {resp.text[:200]}")
            return result
    except Exception as exc:
        result["errors"].append(f"List failed: {exc}")
        return result

    # Test preferred models
    for model in PROVIDERS["GOOGLE"]["preferred_models"]:
        if model not in all_models:
            continue
        try:
            url = PROVIDERS["GOOGLE"]["chat_url"].format(model=model)
            payload = {
                "contents": [{"parts": [{"text": TEST_PROMPT}]}],
                "generationConfig": {"maxOutputTokens": 10, "temperature": 0},
            }
            resp = await client.post(
                url,
                params={"key": api_key},
                json=payload,
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                result["working_models"].append(model)
            else:
                result["errors"].append(f"{model}: HTTP {resp.status_code} - {resp.text[:150]}")
        except Exception as exc:
            result["errors"].append(f"{model}: {exc}")
        await asyncio.sleep(0.2)

    return result


async def test_openai_compatible(
    client: httpx.AsyncClient, provider_name: str, api_key: str
) -> dict[str, Any]:
    """Test OpenAI-compatible providers (Groq, SambaNova, Mistral, NVIDIA, OpenRouter)."""
    config = PROVIDERS[provider_name]
    result = {"provider": provider_name, "models": [], "working_models": [], "errors": []}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # List models
    try:
        resp = await client.get(config["models_url"], headers=headers, timeout=TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("data", data.get("models", []))
            if isinstance(items, list):
                result["models"] = [m.get("id", "") for m in items if isinstance(m, dict) and m.get("id")]
            elif isinstance(data, dict) and "data" not in data:
                # OpenRouter returns {data: [...]}
                pass
        elif resp.status_code == 404:
            # Some providers don't support listing (SambaNova sometimes)
            result["errors"].append("Models list endpoint returned 404; will test preferred models directly")
        else:
            result["errors"].append(f"List failed: HTTP {resp.status_code} - {resp.text[:200]}")
    except Exception as exc:
        result["errors"].append(f"List failed: {exc}")

    # Test preferred models
    models_to_test = list(dict.fromkeys(PROVIDERS[provider_name]["preferred_models"] + result["models"][:15]))
    for model in models_to_test:
        if not model:
            continue
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": TEST_PROMPT}],
                "max_tokens": 10,
                "temperature": 0,
            }
            resp = await client.post(
                config["chat_url"],
                headers=headers,
                json=payload,
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                result["working_models"].append(model)
            elif resp.status_code == 404:
                pass  # model doesn't exist
            else:
                err_text = resp.text[:150]
                result["errors"].append(f"{model}: HTTP {resp.status_code} - {err_text}")
        except Exception as exc:
            result["errors"].append(f"{model}: {exc}")
        await asyncio.sleep(0.3)

    return result


# ─────────────────────────────────────────────────────────────
# Main Orchestration
# ─────────────────────────────────────────────────────────────
async def run_all_tests() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        tasks = []
        for name, config in PROVIDERS.items():
            api_key = ENV.get(config["key_name"], "").strip()
            if not api_key:
                results.append({
                    "provider": name,
                    "models": [],
                    "working_models": [],
                    "errors": ["API key not found in .env"],
                })
                continue
            if name == "GOOGLE":
                tasks.append(test_google(client, api_key))
            else:
                tasks.append(test_openai_compatible(client, name, api_key))

        results.extend(await asyncio.gather(*tasks))
    return results


def print_report(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("  API KEY & MODEL DISCOVERY REPORT")
    print("=" * 80)

    total_working = 0
    routing_recommendations: dict[str, list[str]] = {
        "fast": [],      # for query expansion, simple tasks
        "strong": [],    # for ranking, learning paths
        "judge": [],     # for consensus tie-breaking
        "code": [],      # for code tasks
    }

    for result in sorted(results, key=lambda r: r["provider"]):
        name = result["provider"]
        print(f"\n{'─' * 80}")
        print(f"  {name}")
        print(f"{'─' * 80}")

        key_present = bool(ENV.get(PROVIDERS[name]["key_name"], ""))
        print(f"  API Key:     {'✅ Present' if key_present else '❌ Missing'}")
        print(f"  Models List: {len(result['models'])} available" if result["models"] else "  Models List: (not retrieved)")
        print(f"  Working:     {len(result['working_models'])} verified")

        if result["working_models"]:
            print(f"\n  ✅ VERIFIED WORKING MODELS:")
            for model in result["working_models"]:
                print(f"     • {model}")
                total_working += 1

                # Categorize for routing
                model_lower = model.lower()
                if name == "GOOGLE" and ("flash" in model_lower or "2.5-flash" in model_lower):
                    routing_recommendations["fast"].append((name, model))
                elif name == "GOOGLE" and "pro" in model_lower:
                    routing_recommendations["strong"].append((name, model))
                    routing_recommendations["judge"].append((name, model))
                elif name == "GROQ":
                    routing_recommendations["fast"].append((name, model))
                    if "70b" in model_lower or "405b" in model_lower or "235b" in model_lower:
                        routing_recommendations["strong"].append((name, model))
                elif name == "SAMBANOVA":
                    if "8b" in model_lower or "70b" in model_lower or "405b" in model_lower or "v3" in model_lower or "qwen" in model_lower:
                        routing_recommendations["strong"].append((name, model))
                elif name == "MISTRAL":
                    if "codestral" in model_lower:
                        routing_recommendations["code"].append((name, model))
                    elif "large" in model_lower or "medium" in model_lower or "nemo" in model_lower:
                        routing_recommendations["strong"].append((name, model))
                        routing_recommendations["judge"].append((name, model))
                    elif "small" in model_lower:
                        routing_recommendations["fast"].append((name, model))
                elif name == "NVIDIA":
                    routing_recommendations["strong"].append((name, model))
                elif name == "OPENROUTER":
                    if "flash" in model_lower:
                        routing_recommendations["fast"].append((name, model))
                    else:
                        routing_recommendations["strong"].append((name, model))

        if result["errors"]:
            print(f"\n  ⚠️  ERRORS:")
            for error in result["errors"][:5]:
                print(f"     • {error}")
            if len(result["errors"]) > 5:
                print(f"     ... and {len(result['errors']) - 5} more")

    # ── Routing Recommendations ──
    print(f"\n\n{'=' * 80}")
    print("  ROUTING RECOMMENDATIONS")
    print("=" * 80)
    print(f"\n  Total working models discovered: {total_working}\n")

    for role, models in routing_recommendations.items():
        unique = list(dict.fromkeys(models))[:5]
        print(f"\n  {role.upper()} models (for {role} tasks):")
        if unique:
            for provider, model in unique:
                print(f"     • {provider} / {model}")
        else:
            print(f"     (none found)")

    # ── Suggested routing_rules.yaml ──
    print(f"\n\n{'=' * 80}")
    print("  SUGGESTED configs/routing_rules.yaml")
    print("=" * 80)
    print("\nCopy this into your configs/routing_rules.yaml:\n")

    yaml_lines = ["# Auto-generated routing based on verified working models", ""]
    yaml_lines.append("task_routing:")
    for role in ["fast", "strong", "judge", "code"]:
        yaml_lines.append(f"  {role}:")
        unique = list(dict.fromkeys(routing_recommendations[role]))[:5]
        if unique:
            for i, (provider, model) in enumerate(unique):
                priority = (i + 1) * 10
                yaml_lines.append(f"    - provider: {provider}")
                yaml_lines.append(f"      model_id: \"{model}\"")
                yaml_lines.append(f"      priority: {priority}")
        else:
            yaml_lines.append("    []  # No verified models for this role")

    print("\n".join(yaml_lines))
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    if not ENV:
        print("ERROR: No API keys found. Check your .env file.")
        sys.exit(1)

    print(f"\n🔍 Testing {len(PROVIDERS)} providers...")
    print(f"   Found keys for: {', '.join(name for name, cfg in PROVIDERS.items() if ENV.get(cfg['key_name']))}")
    print(f"   This may take 30-60 seconds...\n")

    start = time.perf_counter()
    results = asyncio.run(run_all_tests())
    elapsed = time.perf_counter() - start

    print(f"\n✓ Completed in {elapsed:.1f}s")
    print_report(results)