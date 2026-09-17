# Research Agent

An **autonomous AI research assistant** that searches the internet and academic databases, ranks the best learning resources using a **multi-model LLM consensus engine**, and builds a structured **learning path** from beginner to advanced.

Given a single topic, the agent gathers papers, code repositories, documentation, and models, then deliberates across multiple AI providers to produce a curated, ranked curriculum — all from the command line.

---

## What It Does

1. **Expands your query** into effective search variants.
2. **Fetches sources in parallel** from seven platforms.
3. **Normalizes, deduplicates, and validates** everything it finds.
4. **Scores sources** with a heuristic engine.
5. **Ranks sources with AI consensus** — multiple LLMs vote, and a judge resolves disagreements.
6. **Builds a step-by-step learning path** grouped by difficulty.
7. **Saves results** as JSON, Markdown, and into a local SQLite database.
8. **Displays a beautiful terminal report** with Rich.

---

## Features

- **Multi-source search:** arXiv, Semantic Scholar, OpenAlex, GitHub, Wikipedia, Hugging Face, and web search.
- **Multi-model consensus ranking:** Google Gemini, DeepSeek, Mistral, Groq, and NVIDIA NIM working together.
- **Automatic difficulty classification:** beginner, intermediate, advanced.
- **Resilient pipeline:** retries, rate limiting, caching, and graceful fallbacks.
- **Persistent history:** every run is stored in SQLite and can be inspected later.
- **Interactive mode:** a guided prompt-driven research session.
- **Zero-trust design:** API keys are loaded from `.env`, never hardcoded, and secrets are masked in logs.

---

## Supported Sources

| Platform          | Type             | Best For                                   |
|-------------------|------------------|--------------------------------------------|
| arXiv             | Research papers  | Cutting-edge preprints                     |
| Semantic Scholar  | Research papers  | Citation data and abstracts                |
| OpenAlex          | Research papers  | Open scholarly catalog                     |
| GitHub            | Repositories     | Code and implementations                   |
| Wikipedia         | Documentation    | Foundational overviews                     |
| Hugging Face      | Models/Datasets  | Pretrained models                          |
| Web (DuckDuckGo)  | General          | Blogs, courses, videos                     |

---

## Installation

### Prerequisites

- Python 3.10 or newer
- API keys for at least one LLM provider (Google, SambaNova, Groq, Mistral, NVIDIA)

### Automated Install (Windows PowerShell)

```powershell
.\scripts\install.ps1
```

This creates a virtual environment, installs dependencies, and prepares a `.env` file.

### Manual Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root:

```text
SAMBANOVA_API_KEY=your_key
GROQ_API_KEY=your_key
GOOGLE_API_KEY=your_key
MISTRAL_API_KEY=your_key
NVIDIA_API_KEY=your_key

GITHUB_TOKEN=optional_key
SEMANTIC_SCHOLAR_API_KEY=optional_key
HUGGINGFACE_TOKEN=optional_key
```

Only the LLM provider keys are required. The platform tokens (GitHub, Hugging Face, Semantic Scholar) are optional but raise rate limits.

Verify your configuration:

```powershell
python main.py config
```

---

## Usage

### One-shot search

```powershell
python main.py search "Autonomous AI" --max-results 10 --limit 5
```

### Search without LLM ranking (fast, no token cost)

```powershell
python main.py search "Autonomous AI" --max-results 5 --limit 5 --no-llm-ranking
```

### Limit to specific platforms

```powershell
python main.py search "Autonomous AI" --platform arxiv --platform github --platform wikipedia
```

### Output as JSON or Markdown

```powershell
python main.py search "Autonomous AI" --json
python main.py search "Autonomous AI" --markdown
```

### Interactive mode

```powershell
python main.py interactive
```

### View run history

```powershell
python main.py runs list
python main.py runs show <request-id>
```

### Show version

```powershell
python main.py version
```

---

## Example Output

```text
╭─────────────────────── Research Agent Report ───────────────────────╮
│          Topic  Autonomous AI                                       │
│           Goal  Learn from basics to advanced                       │
│         Status  SUCCESS                                             │
│        Latency  3642 ms                                             │
│        Sources  10                                                  │
│         Ranked  10                                                  │
│ Learning Steps  4                                                   │
╰─────────────────────────────────────────────────────────────────────╯

  Ranked Sources
  1. Distributing Accountability, Not Capability...   openalex  0.89
  2. Pivotal trial of an autonomous AI-based...       openalex  0.87
  3. Rise of the machines: Delegating decisions...    openalex  0.83

  Learning Path
  Step 1: Build the fundamentals
  Step 2: Strengthen core concepts
  Step 3: Study advanced research
  Step 4: Deepen and explore
```

---

## Project Structure

```text
research_agent/
├── main.py                 Entry point
├── cli.py                  Command-line interface
├── config.py               Settings access
├── constants.py            Global constants
├── exceptions.py           Custom exceptions
├── models.py               Domain models
├── schemas.py              Validation schemas
│
├── clients/                API clients for each platform
├── search/                 Fetch, normalize, dedupe, orchestrate
├── ranking/                Scoring and consensus ranking
├── llm/                    Multi-model provider and ensemble engine
├── storage/                JSON, Markdown, SQLite writers
├── core/                   Pipeline state, events, orchestration
├── utils/                  Retry, rate limit, cache, logging
├── configs/                YAML configuration templates
├── prompts/                Prompt templates
├── tests/                  Unit and integration tests
├── docs/                   Architecture and roadmap
└── scripts/                Install and run helpers
```

---

## Tech Stack

- **Python 3.10+** — core language
- **httpx** — async HTTP client
- **Pydantic v2** — validation and models
- **Typer + Rich** — CLI and terminal UI
- **feedparser** — arXiv feed parsing
- **SQLite** — local run history
- **asyncio** — concurrent fetching and LLM calls

---

## Testing

```powershell
python -m pytest tests/ -v
```

Run static analysis:

```powershell
pyright .
ruff check .
```

---

## How It Stays Reliable

- **Rate limiting** per platform to respect API quotas.
- **Retries with exponential backoff** for transient failures.
- **In-memory caching** to avoid duplicate requests.
- **Heuristic fallback** if all LLM providers fail.
- **Secret masking** so API keys never appear in logs.
- **Graceful degradation** — a failed platform never crashes the pipeline.

---

## License

For personal learning and research use.