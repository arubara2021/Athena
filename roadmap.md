
# Roadmap

This document tracks the evolution of the Research Agent: what has been built, what is stable, what is known to have limitations, and what comes next. It serves as the single source of truth for project progress and future direction.

---

## Table of Contents

1. [Project Status](#project-status)
2. [Completed Phases](#completed-phases)
3. [Stable Capabilities](#stable-capabilities)
4. [Known Limitations](#known-limitations)
5. [Planned Enhancements](#planned-enhancements)
6. [Versioning Plan](#versioning-plan)
7. [Guiding Principles](#guiding-principles)

---

## Project Status

The Research Agent is **feature-complete and production-ready** for personal research use. All twelve development phases have been completed and verified. The system can search across seven platforms concurrently, rank results using multi-model AI consensus, generate structured learning paths, and persist output in three formats.

```
  DEVELOPMENT PROGRESS
  ====================

  Phase  1: Foundation           [##########] COMPLETE
  Phase  2: Utilities            [##########] COMPLETE
  Phase  3: LLM Engine           [##########] COMPLETE
  Phase  4: API Clients          [##########] COMPLETE
  Phase  5: Search Pipeline      [##########] COMPLETE
  Phase  6: Ranking Engine       [##########] COMPLETE
  Phase  7: Storage              [##########] COMPLETE
  Phase  8: Core Pipeline        [##########] COMPLETE
  Phase  9: Interface            [##########] COMPLETE
  Phase 10: Configs and Prompts  [##########] COMPLETE
  Phase 11: Testing              [##########] COMPLETE
  Phase 12: Scripts and Docs     [##########] COMPLETE

  Overall: 12/12 phases complete
  Status:  Feature-complete, production-ready
```

---

## Completed Phases

Each phase built a specific layer of the system. The phases were designed to be sequential and additive, meaning each phase depends only on the phases that came before it.

### Phase 1: Foundation

Established the core building blocks that every other module depends on.

- **Settings**: Centralized configuration management with environment variable loading.
- **Constants**: Shared numeric and string constants used across the codebase.
- **Exceptions**: A hierarchy of custom exception types (`SourceFetchError`, `PipelineError`, `LLMError`, etc.) that provide consistent error handling throughout the system.
- **Domain models**: Pydantic models for `Source`, `RankedSource`, `LearningPath`, `PipelineState`, and other core entities.
- **Schemas**: Validation schemas that enforce the shape of data at every boundary between stages.
- **Structured logging**: A logging configuration that produces consistent, parseable log output with timestamps, severity levels, module names, and request IDs.

### Phase 2: Utilities

Built the shared infrastructure that all layers use for cross-cutting concerns.

- **Retry with backoff**: A decorator that retries failed function calls with exponential backoff and configurable jitter. Supports per-exception retry policies.
- **Async rate limiter**: A token bucket rate limiter that works with Python's asyncio. Each platform and LLM provider gets its own limiter instance.
- **In-memory cache**: A fast, dictionary-based cache for storing API responses within a single run. Entries are keyed by request URL and parameters.
- **Disk cache**: A persistent cache that survives across runs. Used for expensive operations that produce stable results.
- **Hashing**: Utility functions for generating deterministic hashes from URLs, titles, and composite keys. Used by the normalizer and deduplicator.
- **Text helpers**: Functions for cleaning text, truncating strings, normalizing unicode, and extracting keywords.
- **URL helpers**: Functions for normalizing URLs, resolving relative paths, stripping tracking parameters, and generating URL fingerprints.
- **Validators**: Input validation functions for checking URLs, dates, email addresses, and other common data types.
- **Async helpers**: Utility functions for running concurrent tasks with bounded parallelism, collecting results, and handling partial failures.

### Phase 3: LLM Engine

Built the complete language model abstraction layer, from provider management to consensus aggregation.

- **Unified provider manager**: A single async interface over five LLM providers (SambaNova, Groq, Mistral, NVIDIA, Google Gemini). Handles authentication, retries, rate limits, timeouts, and response parsing.
- **JSON parser**: A robust parser that extracts structured JSON from LLM text responses. Handles markdown code fences, mixed text and JSON, trailing commas, and other common formatting issues.
- **Guardrails**: Validation checks on LLM outputs that verify format compliance, content safety, and ranking integrity before the output is consumed by downstream stages.
- **Prompt builders**: Functions that take a template and a set of variables and produce a complete, formatted prompt string.
- **Model router**: A routing engine that selects the appropriate model for each task based on task type, complexity, and availability. Maintains fallback chains for automatic failover.
- **Ensemble engine**: Runs the same prompt against multiple models concurrently and collects their outputs as independent votes.
- **Consensus aggregator**: Compares votes from the ensemble, selects a majority answer when available, delegates to a judge model when there is disagreement, and falls back to heuristic ranking when all LLMs fail.

### Phase 4: API Clients

Built platform-specific clients for all seven data sources.

- **Base client**: A shared `BaseHTTPClient` class that provides connection pooling, rate limiting, retries, caching, and error translation to every platform client.
- **arXiv client**: Queries the arXiv Atom/XML feed. Parses XML responses into the shared `Source` model. No authentication required.
- **Semantic Scholar client**: Queries the Semantic Scholar JSON REST API. Supports optional API key for higher rate limits.
- **OpenAlex client**: Queries the OpenAlex JSON REST API. No authentication required. Provides rich metadata including citation counts and open access status.
- **GitHub client**: Queries the GitHub REST API for repositories, code, and documentation. Supports optional token for higher rate limits.
- **Wikipedia client**: Queries the Wikipedia REST and Action APIs. Extracts article summaries and links. No authentication required.
- **Hugging Face client**: Queries the Hugging Face JSON REST API for models, datasets, and spaces. Supports optional token for higher rate limits.
- **Web search client**: Queries DuckDuckGo's JSON search API for general web results. No authentication required.

### Phase 5: Search Pipeline

Built the orchestration layer that coordinates the entire search phase.

- **Query expansion**: Turns a single topic into multiple targeted queries tailored to each platform. Supports both rule-based and LLM-assisted expansion modes.
- **Concurrent fetching**: Dispatches queries to all seven platform clients simultaneously with a bounded semaphore to prevent API overload.
- **Normalization**: Cleans titles, abstracts, and URLs. Generates stable source IDs from the combination of platform, URL, and title.
- **Deduplication**: Merges duplicate sources using DOI matching, URL fingerprinting, and normalized title comparison. Keeps the richest version of each duplicate.
- **Validation**: Drops invalid sources including empty titles, broken URLs, withdrawn papers, and sources with insufficient metadata.
- **Search orchestrator**: Wires all search sub-stages together, passes data between them, and collects errors without letting any single failure halt the search.

### Phase 6: Ranking Engine

Built the intelligent ranking system that orders sources by relevance and quality.

- **Heuristic scorer**: A fast, dependency-free scorer that evaluates sources on seven dimensions: relevance, platform authority, source type quality, citations, recency, abstract completeness, and code/PDF availability.
- **Difficulty classifier**: Assigns a difficulty level (beginner, intermediate, advanced) to each source based on vocabulary complexity, source type, platform, and presence of code examples.
- **Consensus ranker**: The Mixture-of-Agents engine that sends candidates to multiple LLMs, collects votes, and reconciles disagreements through a judge model.
- **LLM ranker**: The individual ranking call to a single model. Formats candidates into a structured prompt, sends it to the provider, and parses the ranked output.
- **Learning path builder**: Groups ranked sources into difficulty-based tiers and orders them into a structured learning progression from foundational to advanced material.

### Phase 7: Storage

Built the persistence layer that saves results in three complementary formats.

- **File manager**: Coordinates all write operations. Ensures atomic writes (write to temp file, then rename), automatic directory creation, and consistent path generation.
- **JSON writer**: Produces a machine-readable JSON file containing the complete run output: query, sources, ranked results, learning path, errors, and timing metadata.
- **Markdown writer**: Produces a human-readable Markdown report with source summaries, ranked lists, learning paths, and error logs.
- **SQLite store**: Provides persistent storage for run history and source data. Enables cross-run deduplication, historical comparison, and keyword search across all saved sources.

### Phase 8: Core Pipeline

Built the backbone that connects all stages into a single, executable pipeline.

- **Pipeline state**: A mutable object that travels through the pipeline holding the query, sources at every stage, ranked results, learning path, errors, timing data, and run metadata.
- **Event bus**: A publish/subscribe system that allows the CLI and other components to subscribe to pipeline events (search started, ranking completed, etc.) for live progress updates.
- **Main pipeline**: The conductor that runs each stage in order, emits events, collects errors, and classifies the final run as success, partial success, or failure.

### Phase 9: Interface

Built the user-facing command-line interface.

- **Rich-powered CLI**: A terminal interface built with the Rich library that provides formatted tables, progress spinners, syntax highlighting, and color-coded output.
- **Search command**: Run a research query from the command line with a single topic string.
- **Interactive command**: Enter an interactive session where you can run multiple queries, inspect results, and explore learning paths.
- **Runs command**: Browse past runs, view summaries, and compare results across sessions.
- **Config command**: View and modify configuration settings from the command line.
- **Version command**: Display the current version, installed dependencies, and system information.

### Phase 10: Configs and Prompts

Built the configuration and prompt management system.

- **YAML configuration templates**: Six configuration files in `configs/` that control every aspect of the system:
  - `settings.yaml`: General settings (timeouts, output paths, logging level)
  - `models.yaml`: Model definitions and provider assignments
  - `provider_config.yaml`: Provider-specific settings (API base URLs, rate limits)
  - `routing_rules.yaml`: Task-to-model routing rules and fallback chains
  - `ensemble.yaml`: Ensemble configuration (number of voters, judge model, consensus threshold)
  - `sources.yaml`: Platform-specific settings (enabled platforms, per-platform limits)
- **Prompt templates**: Seven prompt template files in `llm/prompts/` that define the instructions sent to LLMs for each task type.

### Phase 11: Testing

Built a comprehensive test suite covering unit, integration, and fixture-based testing.

- **Pytest fixtures**: Shared test fixtures in `conftest.py` that provide mock data, mock clients, and test configuration.
- **Sample data**: JSON fixtures for arXiv and Semantic Scholar responses used in unit tests.
- **Unit tests for clients**: Tests that verify each platform client correctly parses its native API response format into the shared `Source` model.
- **Unit tests for deduplication**: Tests that verify the deduplicator correctly identifies and merges duplicates using DOI, URL, and title matching.
- **Unit tests for ranking**: Tests that verify the heuristic scorer, difficulty classifier, and consensus ranker produce correct and consistent results.
- **Integration tests for the full pipeline**: End-to-end tests that run the entire pipeline from topic to saved output and verify that all stages execute correctly.

### Phase 12: Scripts and Documentation

Built the deployment scripts and project documentation.

- **Install script** (`scripts/install.ps1`): A PowerShell script that creates a virtual environment, installs dependencies, and sets up the `.env` file from the example template.
- **Run script** (`scripts/run_search.ps1`): A PowerShell script that activates the environment and runs a search query.
- **Final dependencies**: A complete and pinned `requirements.txt` and `pyproject.toml` with all direct and transitive dependencies.
- **README**: The project README with overview, installation instructions, usage examples, and configuration guide.
- **Architecture guide**: A detailed document explaining the system's layered design, data flow, and key design decisions.
- **This roadmap**: The project roadmap tracking completed work, current status, and future plans.

---

## Stable Capabilities

The following capabilities have been implemented, tested, and verified as stable:

```
  STABLE CAPABILITIES
  ===================

  +-----------------------------------+        +-------------------------------+
  |  Multi-platform concurrent search |------->|  7 platforms searched in       |
  |                                   |        |  parallel with bounded         |
  |                                   |        |  concurrency                   |
  +-----------------------------------+        +-------------------------------+

  +-----------------------------------+        +-------------------------------+
  |  Multi-model consensus ranking    |------->|  Multiple LLMs vote, judge    |
  |                                   |        |  breaks ties, heuristic        |
  |                                   |        |  fallback always available     |
  +-----------------------------------+        +-------------------------------+

  +-----------------------------------+        +-------------------------------+
  |  Learning path generation         |------->|  Ranked sources grouped into   |
  |                                   |        |  beginner, intermediate, and   |
  |                                   |        |  advanced tiers                |
  +-----------------------------------+        +-------------------------------+

  +-----------------------------------+        +-------------------------------+
  |  Multi-format persistence         |------->|  JSON for machines, Markdown   |
  |                                   |        |  for humans, SQLite for        |
  |                                   |        |  history and querying          |
  +-----------------------------------+        +-------------------------------+

  +-----------------------------------+        +-------------------------------+
  |  Interactive CLI                  |------->|  Rich-powered terminal with    |
  |                                   |        |  live progress, formatted      |
  |                                   |        |  tables, and color output      |
  +-----------------------------------+        +-------------------------------+

  +-----------------------------------+        +-------------------------------+
  |  Run history inspection           |------->|  Browse past runs, compare     |
  |                                   |        |  results, inspect errors       |
  +-----------------------------------+        +-------------------------------+
```

---

## Known Limitations

The system is functional and reliable, but the following limitations are acknowledged and documented.

### Web Search Quality

Web search relies on DuckDuckGo's JSON API, which is free but limited compared to paid search APIs such as Google Custom Search or Bing Web Search. Results may be less comprehensive, less recent, and less relevant for niche or specialized topics. Upgrading to a paid search API would improve web search quality significantly but introduces cost and API key management.

### Platform Rate Limits

Some platforms enforce strict rate limits for unauthenticated requests:

| Platform          | Rate Limit Without Key          | Mitigation                     |
|-------------------|---------------------------------|--------------------------------|
| Semantic Scholar  | 100 requests per 5 minutes      | Optional API key increases     |
|                   |                                 | limit to 1 request per second  |
| GitHub            | 60 requests per hour            | Optional token increases       |
|                   |                                 | limit to 5,000 requests per    |
|                   |                                 | hour                           |
| arXiv             | No hard limit, but aggressive   | Semaphore bounding and         |
|                   | querying triggers throttling    | retry with backoff             |
| OpenAlex          | Polite pool with email,         | Polite email header included   |
|                   | otherwise standard rate         | in all requests                |
| Hugging Face      | Moderate limits without token   | Optional token increases       |
|                   |                                 | limit                          |

The system handles rate limits gracefully through retries and backoff, but sustained high-volume use without API keys will result in slower runs and occasional dropped results.

### arXiv Withdrawn Papers

arXiv occasionally returns papers that have been withdrawn by their authors. The validator filters these based on the withdrawal flag in the arXiv metadata, but edge cases may slip through when the withdrawal metadata is incomplete or delayed. This is a known data quality issue on the arXiv side.

### LLM Consensus Quality

The quality of the AI consensus ranking depends on:

- **Underlying model capability**: Free-tier models may produce less accurate rankings than premium models.
- **Model availability**: Free-tier providers may experience downtime, rate limiting, or degraded performance during peak usage.
- **Prompt sensitivity**: Ranking quality can vary based on how the prompt is worded. The prompt templates have been tested and tuned but may need further refinement for specific domains.

The heuristic fallback ensures the system always produces a result, but the result quality degrades when LLMs are unavailable.

---

## Planned Enhancements

Future work is organized into three tiers based on priority and complexity.

### Near Term

These enhancements are high priority, relatively low complexity, and address immediate usability gaps.

```
  NEAR TERM ROADMAP
  =================

  +---------------------------+     +---------------------------+
  |  Config Loader            |     |  Prompt Loader            |
  |                           |     |                           |
  |  Read YAML templates from |     |  Load prompt templates    |
  |  configs/ at runtime      |     |  from prompts/ instead    |
  |  instead of inline values |     |  of inline strings        |
  +---------------------------+     +---------------------------+
              |                                   |
              v                                   v
  +---------------------------+     +---------------------------+
  |  Embeddings + RAG         |     |  PDF Download and         |
  |                           |     |  Extraction               |
  |  Vector embeddings of     |     |                           |
  |  saved sources so the     |     |  Download arXiv PDFs,     |
  |  agent can answer         |     |  extract text, and use    |
  |  questions about its own  |     |  full paper content for   |
  |  research library         |     |  ranking and summaries    |
  +---------------------------+     +---------------------------+
```

**Config Loader**: A runtime configuration loader that reads the YAML templates in `configs/` and makes their values available to all modules. Currently, some configuration values are defined inline in code. Moving them to YAML files makes the system easier to configure without modifying source code.

**Prompt Loader**: A runtime prompt loader that reads template files from `prompts/` and `llm/prompts/` and makes them available to the LLM layer. Currently, some prompts are defined as inline strings in the code. Moving them to template files makes them easier to iterate on and version control.

**Embeddings + RAG**: A retrieval-augmented generation system that creates vector embeddings of all saved sources and stores them in a local vector database. This enables the agent to answer natural language questions about the research it has already conducted, such as "What papers did I find about transformer architectures?" or "Summarize the key findings from my last research session."

**PDF Download and Extraction**: The ability to download full PDF files from arXiv and other sources, extract their text content, and use that content for more accurate ranking, summarization, and difficulty classification. Currently, the system works primarily with abstracts and metadata.

### Medium Term

These enhancements are moderate priority and require more significant development effort.

```
  MEDIUM TERM ROADMAP
  ===================

  +---------------------------+     +---------------------------+
  |  Web Dashboard            |     |  NotebookLM-style Chat    |
  |                           |     |                           |
  |  Browse run history and   |     |  Conversational interface |
  |  learning paths in a      |     |  over your saved research |
  |  visual web interface     |     |  library                  |
  +---------------------------+     +---------------------------+
              |                                   |
              v                                   v
  +---------------------------+     +---------------------------+
  |  Scheduled Research       |     |  Export Integrations      |
  |                           |     |                           |
  |  Automatically re-run     |     |  Export results to Notion |
  |  topics on a schedule     |     |  and Obsidian in their    |
  |  and alert when new       |     |  native formats           |
  |  papers are found         |     |                           |
  +---------------------------+     +---------------------------+
```

**Web Dashboard**: A browser-based interface for browsing run history, exploring learning paths, comparing results across sessions, and visualizing source distributions by platform, difficulty, and topic. The dashboard would read from the SQLite database and present the data in an interactive format.

**NotebookLM-style Chat**: A conversational interface that allows the user to ask natural language questions about their saved research library. Built on top of the Embeddings + RAG system from the near-term plan, this feature would provide a chat experience similar to Google's NotebookLM, where the user can explore, summarize, and synthesize their research through dialogue.

**Scheduled Research**: The ability to configure recurring research runs that automatically re-examine topics on a daily or weekly schedule. When new papers or sources are found that were not present in previous runs, the system sends an alert (email, webhook, or desktop notification). This feature turns the system from a one-time research tool into a continuous research monitor.

**Export to Notion and Obsidian**: Native export integrations that convert research results into formats compatible with Notion and Obsidian. For Notion, this means creating pages with databases. For Obsidian, this means creating interlinked Markdown files with proper frontmatter and backlinks.

### Long Term

These enhancements are ambitious, high-impact features that require significant research and development.

```
  LONG TERM ROADMAP
  =================

  +---------------------------+     +---------------------------+
  |  Autonomous Deep Reading  |     |  Citation Graph Traversal |
  |                           |     |                           |
  |  Download and summarize   |     |  Discover foundational    |
  |  full papers, not just    |     |  and follow-up work by    |
  |  abstracts                |     |  traversing citation       |
  |                           |     |  networks                  |
  +---------------------------+     +---------------------------+
              |                                   |
              v                                   v
  +---------------------------+     +---------------------------+
  |  Multi-language Support   |     |  Local Model Support      |
  |                           |     |                           |
  |  Search and process       |     |  Run the full pipeline    |
  |  sources in non-English   |     |  with local models via    |
  |  languages                |     |  Ollama for fully         |
  |                           |     |  offline operation         |
  +---------------------------+     +---------------------------+
```

**Autonomous Deep Reading**: The ability to download full papers, read them in their entirety, and produce detailed summaries that capture methodology, results, limitations, and key contributions. This goes beyond abstract-based analysis and requires advanced document understanding capabilities including table extraction, figure interpretation, and mathematical notation parsing.

**Citation Graph Traversal**: The ability to follow citation networks in both directions. Given a paper, the system would identify the foundational papers it cites (backward traversal) and the follow-up papers that cite it (forward traversal). This enables the discovery of seminal works, research lineages, and emerging trends that are not visible from individual paper metadata alone.

**Multi-language Support**: The ability to search for and process sources written in languages other than English. This includes querying non-English databases, translating titles and abstracts, and assessing the relevance of non-English sources to the original query.

**Local Model Support via Ollama**: The ability to run the entire LLM layer using locally hosted models through Ollama. This would enable fully offline operation with no API keys, no rate limits, and no data leaving the local machine. The tradeoff is reduced model quality compared to cloud-hosted frontier models, but for many research tasks, local models would be sufficient.

---

## Versioning Plan

The project follows semantic versioning. Each version corresponds to a set of completed features.

```
  VERSION TIMELINE
  ================

  v0.1.0  [################]  CURRENT
          |
          |  Initial autonomous research pipeline
          |  All 12 phases complete
          |  Multi-platform search, consensus ranking,
          |  learning paths, multi-format persistence,
          |  interactive CLI, run history
          |
          v
  v0.2.0  [................]  PLANNED
          |
          |  Config loader and prompt loader
          |  Embeddings and RAG
          |  PDF download and extraction
          |
          v
  v0.3.0  [................]  PLANNED
          |
          |  Web dashboard
          |  Export integrations (Notion, Obsidian)
          |
          v
  v0.4.0  [................]  PLANNED
          |
          |  Scheduled research
          |  NotebookLM-style chat
          |
          v
  v1.0.0  [................]  FUTURE
          |
          |  Autonomous deep reading
          |  Citation graph traversal
          |  Multi-language support
          |  Local model support
          |  Production-grade stability and documentation
```

| Version | Status   | Key Features                                                      |
|---------|----------|-------------------------------------------------------------------|
| v0.1.0  | Current  | Full pipeline, 7 platforms, consensus ranking, learning paths,    |
|         |          | JSON/Markdown/SQLite persistence, interactive CLI, run history    |
| v0.2.0  | Planned  | Config/prompt loaders, embeddings + RAG, PDF extraction           |
| v0.3.0  | Planned  | Web dashboard, Notion/Obsidian export                             |
| v0.4.0  | Planned  | Scheduled research, NotebookLM-style chat                         |
| v1.0.0  | Future   | Deep reading, citation graphs, multi-language, local models       |

---

## Guiding Principles

Every design decision, feature addition, and architectural choice is evaluated against these five principles. They are listed in priority order.

### 1. Accuracy Over Speed

The system exists to help researchers find the best sources for a given topic. Getting the right answer slowly is always preferred over getting the wrong answer quickly. This principle drives the consensus ranking approach, where multiple models vote on relevance rather than relying on a single fast model. It also drives the heuristic fallback, which prioritizes correctness through multi-dimensional scoring over raw throughput.

### 2. Resilience

No single failure should crash a run. Every stage catches its own errors. Every network call has a timeout and a retry policy. Every LLM call has a fallback. Every file write is atomic. The system is designed to operate in imperfect conditions: flaky APIs, rate-limited providers, intermittent network connectivity, and incomplete data. It degrades gracefully rather than failing catastrophically.

### 3. Transparency

Every decision the system makes is logged and every run is saved. The user can always answer the following questions:

- What queries were sent to which platforms?
- Which sources were found and which were filtered?
- How did the ranking models vote and what was the final consensus?
- What errors occurred and how were they handled?
- How long did each stage take?

This transparency builds trust and enables debugging. The structured logger, the event bus, and the multi-format persistence layer all exist to serve this principle.

### 4. Extensibility

New platforms and new models should plug in without rewrites. The client layer uses inheritance so that adding a new platform means writing one new class that implements three methods: build request, parse response, and map to Source. The LLM layer uses a provider abstraction so that adding a new model provider means implementing one new provider class. The pipeline uses a stage-based architecture so that adding a new processing step means writing one new stage function and registering it in the pipeline.

### 5. Privacy

API keys stay local and are never logged. The `.env` file is git-ignored. The logger masks sensitive values. No data is sent to third-party analytics services. The SQLite database stays on the local machine. The system is designed for personal use and treats the user's research data as private by default.
