```markdown
# Architecture

The Research Agent is an autonomous, multi-platform research pipeline that takes a single topic, searches across seven academic and technical platforms concurrently, ranks results using a multi-model AI consensus engine, builds a structured learning path, and persists everything in JSON, Markdown, and SQLite. Every stage is independent, testable, and fault-tolerant.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Layered Architecture](#layered-architecture)
3. [Client Layer](#client-layer)
4. [Search Layer](#search-layer)
5. [Ranking Layer](#ranking-layer)
6. [LLM Layer](#llm-layer)
7. [Model Routing](#model-routing)
8. [Core Layer](#core-layer)
9. [Storage Layer](#storage-layer)
10. [End-to-End Sequence](#end-to-end-sequence)
11. [Key Design Decisions](#key-design-decisions)
12. [Reliability Mechanisms](#reliability-mechanisms)

---

## Pipeline Overview

The system operates as a seven-stage linear pipeline. Each stage receives input from the previous one, performs its work, and passes structured output forward. If any stage encounters a failure, it captures the error, logs it, and allows the pipeline to continue with whatever data it has collected so far.

```
                         RESEARCH AGENT PIPELINE
                         =======================

                              [ Topic ]
                                  |
                                  v
                 +---------------------------------+
                 |  1. QUERY EXPANSION             |
                 |     Expand topic into multiple  |
                 |     targeted search queries     |
                 +---------------------------------+
                                  |
                                  v
                 +---------------------------------+
                 |  2. SOURCE FETCHING             |
                 |     Query 7 platforms           |
                 |     concurrently                |
                 +---------------------------------+
                                  |
                                  v
                 +---------------------------------+
                 |  3. NORMALIZATION               |
                 |     Clean, unify, and map all   |
                 |     results to a shared schema  |
                 +---------------------------------+
                                  |
                                  v
                 +---------------------------------+
                 |  4. DEDUPLICATION               |
                 |     Merge duplicates using DOI, |
                 |     URL fingerprint, and title  |
                 +---------------------------------+
                                  |
                                  v
                 +---------------------------------+
                 |  5. VALIDATION                  |
                 |     Drop invalid, withdrawn,    |
                 |     or broken sources           |
                 +---------------------------------+
                                  |
                                  v
                 +---------------------------------+
                 |  6. RANKING                     |
                 |     Heuristic scoring followed  |
                 |     by multi-model AI consensus |
                 +---------------------------------+
                                  |
                                  v
                 +---------------------------------+
                 |  7. LEARNING PATH               |
                 |     Group ranked sources into   |
                 |     difficulty-based steps      |
                 +---------------------------------+
                                  |
                                  v
                 +---------------------------------+
                 |  SAVE                           |
                 |     JSON + Markdown + SQLite    |
                 |     Terminal report rendered    |
                 +---------------------------------+
```

Each stage is a self-contained module with its own error handling, logging, and timeout management. A failure in normalization, for example, will never crash the ranking stage. The pipeline tracks the state of every stage and reports partial successes honestly.

---

## Layered Architecture

The system is organized into eight layers, stacked from user-facing at the top to foundational utilities at the bottom. Each layer depends only on the layer directly beneath it. No layer reaches two levels down. This constraint keeps the dependency graph clean and makes every layer independently testable.

```
+================================================================+
|                                                                |
|   LAYER 1: INTERFACE                                          |
|   cli.py, main.py                                             |
|   User interaction, command parsing, terminal rendering        |
|                                                                |
+================================================================+
                              |
                              v
+================================================================+
|                                                                |
|   LAYER 2: CORE                                               |
|   pipeline.py, state.py, events.py                            |
|   Orchestration, mutable state, pub/sub event bus             |
|                                                                |
+================================================================+
                              |
                              v
+================================================================+
|                                                                |
|   LAYER 3: RANKING                                            |
|   scorer.py, consensus_ranker.py, llm_ranker.py,             |
|   difficulty_classifier.py, learning_path_builder.py          |
|   Heuristic and AI-driven ranking, path generation            |
|                                                                |
+================================================================+
                              |
                              v
+================================================================+
|                                                                |
|   LAYER 4: LLM                                                |
|   provider.py, ensemble.py, aggregator.py,                    |
|   guardrails.py, parser.py, router.py, prompts.py             |
|   Multi-provider abstraction, ensemble voting, aggregation    |
|                                                                |
+================================================================+
                              |
                              v
+================================================================+
|                                                                |
|   LAYER 5: SEARCH                                             |
|   orchestrator.py, query_expander.py, source_fetcher.py,     |
|   normalizer.py, dedupe.py, validator.py                      |
|   Query expansion, concurrent fetching, data cleaning         |
|                                                                |
+================================================================+
                              |
                              v
+================================================================+
|                                                                |
|   LAYER 6: CLIENTS                                            |
|   base_client.py, arxiv_client.py, semantic_scholar_client.py,|
|   openalex_client.py, github_client.py, wikipedia_client.py,  |
|   huggingface_client.py, web_search_client.py                 |
|   Platform-specific API integration                           |
|                                                                |
+================================================================+
                              |
                              v
+================================================================+
|                                                                |
|   LAYER 7: STORAGE                                            |
|   file_manager.py, json_writer.py, markdown_writer.py,       |
|   sqlite_store.py                                             |
|   Atomic file writes, persistent history                      |
|                                                                |
+================================================================+
                              |
                              v
+================================================================+
|                                                                |
|   LAYER 8: UTILITIES                                          |
|   retry.py, rate_limiter.py, cache.py, hashing.py,           |
|   text.py, url.py, validators.py, async_helpers.py,          |
|   logger.py                                                   |
|   Cross-cutting concerns used by every layer                  |
|                                                                |
+================================================================+
```

The Utilities layer is the only layer that is imported by all other layers. It contains no business logic, only shared infrastructure: retry decorators, rate limiters, caches, hash functions, text normalizers, URL validators, async helpers, and the structured logger.

---

## Client Layer

The Client Layer is responsible for communicating with external platforms. Every platform has its own client class that inherits from `BaseHTTPClient`. The base class provides all the shared infrastructure so that each concrete client only needs to know three things: how to build a request, how to parse the response, and how to map the result into the shared `Source` model.

### BaseHTTPClient

The base client provides the following capabilities to every platform client:

- **Connection pooling**: Uses `httpx.AsyncClient` with persistent connections to avoid the overhead of opening a new TCP connection for every request.
- **Per-platform rate limiting**: Each client has its own rate limiter instance so that a burst of requests to one platform does not throttle another.
- **Retries with exponential backoff**: Failed requests are retried up to a configurable number of times. The delay between retries grows exponentially with added jitter to avoid thundering herd problems.
- **In-memory response caching**: Identical requests within the same run are served from cache, reducing API calls and latency.
- **Consistent error translation**: All HTTP errors, timeouts, and parsing failures are caught and re-raised as `SourceFetchError` with a standardized structure. The rest of the system never needs to handle platform-specific exceptions.

### Platform Clients

| Client                  | Platform          | Endpoint Style     | Authentication    | Rate Limit Concern |
|-------------------------|-------------------|--------------------|-------------------|--------------------|
| `ArxivClient`           | arXiv             | Atom/XML feed      | None              | Low                |
| `SemanticScholarClient` | Semantic Scholar  | JSON REST          | Optional API key  | High without key   |
| `OpenAlexClient`        | OpenAlex          | JSON REST          | None              | Moderate           |
| `GithubClient`          | GitHub            | JSON REST          | Optional token    | High without token |
| `WikipediaClient`       | Wikipedia         | REST + Action API  | None              | Low                |
| `HuggingFaceClient`     | Hugging Face      | JSON REST          | Optional token    | Moderate           |
| `WebSearchClient`       | DuckDuckGo        | DuckDuckGo JSON    | None              | Moderate           |

Each client maps its platform-specific response format into the unified `Source` schema. The `Source` model contains fields for title, abstract, URL, platform, source type, authors, citations, publication date, and metadata. This normalization happens at the client level so that every downstream stage works with a single, consistent data structure.

### Client Interaction Flow

```
  Search Layer
       |
       |  async fetch(query, platform)
       v
+------------------+
|  BaseHTTPClient  |
|                  |
|  1. Check cache  |
|  2. Acquire rate |
|     limiter slot |
|  3. Build request|
|  4. Send via     |
|     httpx        |
|  5. Retry on     |
|     failure      |
|  6. Parse resp.  |
|  7. Map to Source|
|  8. Cache result |
+------------------+
       |
       v
  List[Source]
```

---

## Search Layer

The Search Layer coordinates the entire search phase. It is the first major stage in the pipeline and is responsible for turning a raw topic string into a clean, validated, deduplicated list of sources.

### Orchestrator (`orchestrator.py`)

The orchestrator is the conductor of the search phase. It calls each sub-stage in sequence:

1. Query Expansion
2. Source Fetching
3. Normalization
4. Deduplication
5. Validation

It does not perform any logic itself. Its job is to wire the stages together, pass data between them, and collect errors from each stage without letting any single failure halt the entire search.

### Query Expander (`query_expander.py`)

The query expander takes a single topic string and produces multiple targeted search queries. Each expanded query is tailored to a specific platform or content type.

For example, given the topic `Autonomous AI`, the expander might produce:

- `Autonomous AI research paper` (for arXiv, Semantic Scholar, OpenAlex)
- `Autonomous AI github` (for GitHub)
- `Autonomous AI tutorial` (for Wikipedia, Hugging Face)
- `Autonomous AI latest developments` (for web search)

The expander can operate in two modes:

- **Rule-based mode**: Uses templates and heuristics to append platform-specific suffixes. This mode has zero latency and no external dependencies.
- **LLM-assisted mode**: Sends the topic to a fast model (such as `gemini-2.5-flash`) and asks it to generate optimized queries for each platform. This mode produces smarter queries but adds a small amount of latency.

The mode is configurable and the system falls back to rule-based expansion if the LLM is unavailable.

### Source Fetcher (`source_fetcher.py`)

The source fetcher takes the list of expanded queries and dispatches them to all seven platform clients concurrently. It uses a bounded semaphore to limit the number of simultaneous requests, preventing the system from overwhelming any single API.

```
  Expanded Queries
       |
       v
+---------------------+
|   Source Fetcher     |
|                     |
|   Semaphore(n=10)   |
|                     |
|   +-- ArxivClient   |
|   +-- SemanticSch.  |
|   +-- OpenAlex      |
|   +-- GitHub        |
|   +-- Wikipedia     |
|   +-- HuggingFace   |
|   +-- WebSearch     |
|                     |
|   All run in        |
|   parallel with     |
|   bounded concurrency|
+---------------------+
       |
       v
  List[Source] (raw, unnormalized)
```

Each client runs independently. If one platform times out or returns an error, the fetcher captures the error and continues collecting results from the other six platforms. The final output is a flat list of all raw sources returned by all platforms.

### Normalizer (`normalizer.py`)

The normalizer receives the raw list of sources and performs the following cleaning operations:

- **Title cleaning**: Strips whitespace, removes special characters, normalizes unicode, and collapses multiple spaces.
- **Abstract cleaning**: Truncates excessively long abstracts, removes HTML tags, and normalizes line breaks.
- **URL normalization**: Ensures consistent URL formatting, resolves relative URLs, and strips tracking parameters.
- **Source ID generation**: Creates a stable, deterministic `source_id` by hashing the combination of platform name, URL, and title. This ID is used downstream for deduplication and storage.

### Deduplicator (`dedupe.py`)

The deduplicator identifies and merges duplicate sources that appear across multiple platforms. It uses a three-tier matching strategy:

1. **DOI match**: If two sources share the same DOI, they are the same paper. This is the strongest signal.
2. **URL fingerprint match**: URLs are normalized and hashed. Matching fingerprints indicate the same resource.
3. **Normalized title match**: Titles are lowercased, stripped of punctuation, and compared. A high similarity score indicates a duplicate.

When two sources are identified as duplicates, the deduplicator keeps the richest version. Richness is determined by:

- Length of the abstract (longer is better)
- Number of citations or stars (higher is better)
- Number of authors (more complete metadata is better)
- Availability of code or PDF links

The deduplicator merges metadata from both sources when possible, combining author lists and keeping the most complete abstract.

### Validator (`validator.py`)

The validator is the final gate before sources enter the ranking stage. It drops any source that fails basic quality checks:

- Empty or missing title
- Broken or malformed URL
- Marked as withdrawn or retracted (especially relevant for arXiv)
- Abstract is empty or below a minimum length threshold
- Publication date is unreasonably old or in the future

The validator logs every dropped source with the reason for rejection, so the user can inspect what was filtered and why.

---

## Ranking Layer

The Ranking Layer takes the clean, validated list of sources and produces a ranked ordering from most to least relevant. It uses a two-phase approach: a fast heuristic scorer followed by a multi-model AI consensus engine.

### Heuristic Scorer (`scorer.py`)

The heuristic scorer is a fast, dependency-free ranking function that evaluates every source on seven weighted dimensions:

| Dimension             | What It Measures                                    | Weight  |
|-----------------------|-----------------------------------------------------|---------|
| Relevance             | Keyword overlap between query and title/abstract    | Highest |
| Platform authority    | Trust score assigned to each platform               | High    |
| Source type quality   | Papers > tutorials > blog posts > raw web results   | High    |
| Citations / Stars     | Number of citations, GitHub stars, or downloads     | Medium  |
| Recency               | How recently the source was published or updated    | Medium  |
| Abstract completeness | Whether a meaningful abstract is present             | Low     |
| Code / PDF availability | Whether a PDF or code repository is linked        | Low     |

Each dimension produces a normalized score between 0 and 1. The weighted sum produces a final heuristic score for each source. This scorer always runs and serves as the fallback if the LLM layer is completely unavailable.

### Difficulty Classifier (`difficulty_classifier.py`)

The difficulty classifier assigns a difficulty level to each source: beginner, intermediate, or advanced. It evaluates:

- The vocabulary complexity of the title and abstract
- The source type (a survey paper is more accessible than a technical report)
- The platform (Hugging Face tutorials tend to be more beginner-friendly than arXiv papers)
- Whether the source includes code examples or visual explanations

The classifier can operate in rule-based mode or use an LLM for more nuanced classification.

### Consensus Ranker (`consensus_ranker.py`)

The consensus ranker is the intelligent core of the ranking system. It implements a **Mixture-of-Agents** pattern where multiple LLMs independently evaluate the same set of candidate sources and then reconcile their judgments.

The process works as follows:

```
  Top N Candidates (from heuristic scorer)
       |
       v
+------------------------------------------+
|         ENSEMBLE ENGINE                   |
|                                          |
|   +------------+  +------------+         |
|   |  Model A   |  |  Model B   |         |
|   | (Gemini    |  | (DeepSeek  |         |
|   |  2.5 Pro)  |  |  V3)       |         |
|   +------------+  +------------+         |
|          |              |                |
|          v              v                |
|     Vote A         Vote B               |
|          |              |                |
|          +------+-------+                |
|                 |                        |
|                 v                        |
|   +----------------------------+         |
|   |       AGGREGATOR           |         |
|   |                            |         |
|   |  Majority agree?           |         |
|   |    YES -> Use majority     |         |
|   |    NO  -> Send all votes   |         |
|   |          to Judge Model    |         |
|   |          (Gemini 2.5 Pro)  |         |
|   |                            |         |
|   |  Judge fails?              |         |
|   |    -> Fall back to         |         |
|   |       heuristic ranking    |         |
|   +----------------------------+         |
|                 |                        |
|                 v                        |
|        Final Ranked List                 |
+------------------------------------------+
```

**Step 1: Ensemble Voting**

The ensemble engine sends the same ranking prompt to multiple models concurrently. Each model receives the list of candidate sources and is asked to rank them by relevance to the original query. Each model's output is treated as an independent vote.

**Step 2: Aggregation**

The aggregator compares the votes:

- **Clear majority**: If most models agree on the top ranking, that answer wins immediately.
- **Disagreement**: If the models produce conflicting rankings, a designated judge model reads all the votes and selects the best ranking. The judge has access to the full reasoning of each voter.
- **Total failure**: If all LLM calls fail (rate limits, timeouts, errors), the system falls back to the heuristic ranking. The user still gets a result.

This three-tier fallback ensures that the system always produces a ranked output, regardless of LLM availability.

### LLM Ranker (`llm_ranker.py`)

The LLM ranker is the individual ranking call made to a single model. It formats the candidates into a structured prompt, sends it to the provider manager, parses the response, and returns a ranked list. The consensus ranker calls multiple instances of the LLM ranker in parallel.

### Learning Path Builder (`learning_path_builder.py`)

After ranking is complete, the learning path builder groups the ranked sources into a structured learning progression. It organizes sources into difficulty tiers:

- **Beginner**: Introductory tutorials, overview papers, getting-started guides
- **Intermediate**: Detailed implementations, benchmark papers, framework documentation
- **Advanced**: Cutting-edge research, technical reports, original source code

Each tier is ordered so that a learner can progress from foundational concepts to advanced material without gaps. The builder uses the difficulty classifier's output and the consensus ranking to determine placement.

---

## LLM Layer

The LLM Layer provides a unified abstraction over multiple language model providers. It hides the differences between provider APIs, handles authentication, manages rate limits, and provides a clean async interface to the rest of the system.

### Provider Manager (`provider.py`)

The provider manager is a unified async interface over five LLM providers. It abstracts away the API differences between them:

| Provider Category       | Providers                          | API Style              |
|-------------------------|------------------------------------|------------------------|
| OpenAI-compatible       | SambaNova, Groq, Mistral, NVIDIA  | `/chat/completions`    |
| Google-native           | Google Gemini                      | `generateContent`      |

The provider manager handles:

- **Authentication**: Loads API keys from `.env` and attaches them to requests. Keys are never logged or exposed in error messages.
- **Retries**: Failed requests are retried with exponential backoff. Transient errors (429, 500, 503) trigger retries. Permanent errors (401, 403) are raised immediately.
- **Rate limiting**: Each provider has its own rate limiter to stay within free-tier or paid-tier limits.
- **Response parsing**: Raw API responses are parsed into a standardized format regardless of the provider. The rest of the system never sees provider-specific response structures.
- **Timeout management**: Every LLM call has a configurable timeout. If a model does not respond in time, the call is cancelled and the error is captured.

### Ensemble Engine (`ensemble.py`)

The ensemble engine runs the same prompt against multiple models concurrently. It collects each model's output as an independent vote. The engine uses `asyncio.gather` with error isolation so that one model's failure does not affect the others.

```
  Ranking Prompt
       |
       +------------------+------------------+
       |                  |                  |
       v                  v                  v
  +---------+        +---------+        +---------+
  | Model A |        | Model B |        | Model C |
  | Vote    |        | Vote    |        | Vote    |
  +---------+        +---------+        +---------+
       |                  |                  |
       +------------------+------------------+
                          |
                          v
                    List[Votes]
                          |
                          v
                     Aggregator
```

### Aggregator (`aggregator.py`)

The aggregator receives the list of votes and determines the final answer:

1. **Majority consensus**: If more than half the models agree, the majority answer is selected. This is the fastest and most common path.
2. **Judge model**: If there is no clear majority, all votes are sent to a designated judge model. The judge reads each vote, evaluates the reasoning, and selects the best answer. The judge model is typically the strongest available model (such as `gemini-2.5-pro`).
3. **Heuristic fallback**: If every LLM call fails, the aggregator returns the heuristic ranking as the final result.

This layered approach ensures that accuracy is maximized when LLMs are available, and the system degrades gracefully when they are not.

### Guardrails (`guardrails.py`)

The guardrails module validates LLM outputs before they are used by the rest of the system. It checks:

- Output format compliance (valid JSON, expected fields present)
- Content safety (no harmful, irrelevant, or nonsensical output)
- Ranking integrity (no duplicate entries, no missing sources, correct ordering)

If a guardrail check fails, the output is discarded and the system falls back to the next available ranking source.

### Parser (`parser.py`)

The parser extracts structured data from raw LLM text responses. LLMs sometimes wrap their output in markdown code blocks, add explanatory text before or after the JSON, or produce slightly malformed JSON. The parser handles all of these cases:

- Strips markdown code fences
- Extracts JSON from mixed text
- Attempts repair of common JSON errors (trailing commas, missing quotes)
- Validates the parsed structure against the expected schema

### Prompts (`prompts.py`)

The prompts module manages all prompt templates used by the LLM layer. Templates are stored as separate `.txt` files in the `llm/prompts/` directory and are loaded at runtime. Each template uses placeholder variables that are filled in with the actual data for each request.

Available prompt templates:

| Template File               | Purpose                                      |
|-----------------------------|----------------------------------------------|
| `ranking_prompt.txt`        | Ask a model to rank candidate sources        |
| `consensus_judge.txt`       | Ask the judge to pick the best ranking       |
| `difficulty_prompt.txt`     | Classify source difficulty level             |
| `learning_path_prompt.txt`  | Generate a structured learning path          |
| `router_prompt.txt`         | Select the best model for a given task       |
| `aggregator_prompt.txt`     | Aggregate multiple outputs into one          |
| `source_summary_prompt.txt` | Generate a summary for a single source       |

---

## Model Routing

The model router selects the appropriate model for each task based on the task's complexity, latency requirements, and accuracy needs. Different tasks have different requirements: query expansion needs speed, while consensus judging needs the strongest available model.

| Task                  | Model Class | Example Models                | Rationale                                    |
|-----------------------|-------------|-------------------------------|----------------------------------------------|
| Query expansion       | fast        | gemini-2.5-flash              | Speed matters more than nuance               |
| Source ranking        | strong      | gemini-2.5-pro, DeepSeek V3   | Requires nuanced understanding of relevance  |
| Consensus judging     | judge       | gemini-2.5-pro                | Needs the strongest model to break ties      |
| Learning path         | strong      | gemini-2.5-pro                | Requires understanding of topic progression  |
| Code tasks            | code        | codestral-latest              | Specialized for code generation and analysis |

The router maintains fallback chains for each model class. If the primary model is rate-limited or unavailable, the router automatically tries the next model in the chain. This happens transparently; the rest of the system does not need to know which model actually served the request.

```
  Task Request
       |
       v
+------------------+
|  Model Router    |
|                  |
|  1. Identify     |
|     task class   |
|  2. Select       |
|     primary model|
|  3. Check        |
|     availability |
|  4. Fall through |
|     chain if     |
|     needed       |
|  5. Return       |
|     provider ref |
+------------------+
       |
       v
  Provider Manager -> Model
```

---

## Core Layer

The Core Layer is the backbone of the system. It contains the pipeline conductor, the mutable state object, and the event bus.

### Pipeline (`pipeline.py`)

The pipeline is the conductor that runs each stage in order. It is responsible for:

- **Stage execution**: Calling each stage with the current state and collecting the output.
- **Event emission**: Publishing events before and after each stage so that the CLI and other subscribers can track progress.
- **Error collection**: Capturing errors from each stage without letting them propagate and crash the run.
- **Result classification**: After all stages complete, the pipeline classifies the run as one of:
  - **Success**: All stages completed without errors.
  - **Partial success**: Some stages had errors but usable results were still produced.
  - **Failure**: Critical stages failed and no usable results could be produced.
- **Timing**: Recording the duration of each stage and the total run time.

### State (`state.py`)

The state is a single mutable object that travels through the entire pipeline. It holds:

- **Query**: The original topic string and all expanded queries.
- **Sources**: The raw, normalized, deduplicated, and validated source lists at each stage.
- **Ranked results**: The final ranked list produced by the consensus engine.
- **Learning path**: The structured learning path grouped by difficulty.
- **Errors**: A list of all errors encountered during the run, tagged by stage.
- **Timing**: Start time, end time, and per-stage durations.
- **Metadata**: Run ID, timestamp, configuration snapshot, and model versions used.

The state object is passed by reference through every stage. Each stage reads from and writes to the state. After the run completes, the state is serialized to JSON and SQLite for persistence.

### Events (`events.py`)

The events module implements a publish/subscribe bus. Any component can publish an event, and any component can subscribe to events it cares about.

The CLI subscribes to the following events to update the live progress display:

| Event                   | Published When                        | Payload                  |
|-------------------------|---------------------------------------|--------------------------|
| `search_started`        | Search phase begins                   | Query, expanded queries  |
| `search_completed`      | Search phase finishes                 | Source count, errors     |
| `ranking_started`       | Ranking phase begins                  | Candidate count          |
| `ranking_completed`     | Ranking phase finishes                | Ranked list, errors      |
| `learning_path_built`   | Learning path generation completes    | Path structure           |
| `storage_completed`     | All writes finish                     | File paths, DB status    |
| `pipeline_completed`    | Entire run finishes                   | Classification, timing   |
| `error_occurred`        | Any stage encounters an error         | Stage name, error detail |

The event bus is synchronous within a single run. Events are processed in the order they are published.

---

## Storage Layer

The Storage Layer persists the results of every run in three formats, each serving a different purpose.

### JSON Writer (`json_writer.py`)

Produces a machine-readable JSON file containing the complete run output: query, all sources (raw and processed), ranked results, learning path, errors, and timing metadata. This file can be consumed by other tools, scripts, or dashboards.

### Markdown Writer (`markdown_writer.py`)

Produces a human-readable Markdown report. This report includes:

- The original query and expanded queries
- A summary of sources found per platform
- The ranked list with titles, abstracts, URLs, and scores
- The learning path organized by difficulty tier
- Any errors or warnings encountered during the run

This is the file the user reads directly.

### SQLite Store (`sqlite_store.py`)

Provides persistent storage for run history and source data. The database enables:

- **Run history inspection**: Browse past runs, compare results over time.
- **Source deduplication across runs**: Identify sources that appear in multiple research sessions.
- **Querying and filtering**: Search across all saved sources by platform, date, or keyword.

### File Manager (`file_manager.py`)

The file manager coordinates all write operations. It ensures:

- **Atomic writes**: Data is written to a temporary file first, then renamed to the final path. If the process crashes mid-write, the original file is never corrupted.
- **Directory creation**: Output directories are created automatically if they do not exist.
- **Path generation**: File names include the sanitized topic and a timestamp for easy identification.

```
  Pipeline Output
       |
       +------------------+------------------+
       |                  |                  |
       v                  v                  v
  +---------+        +-----------+        +---------+
  |  JSON   |        | Markdown  |        | SQLite  |
  | Writer  |        |  Writer   |        |  Store  |
  +---------+        +-----------+        +---------+
       |                  |                  |
       v                  v                  v
  output/json/       output/markdown/     data/
  topic_ts.json      topic_ts.md          research_agent.db
```

---

## End-to-End Sequence

The following diagram shows the complete interaction flow from the moment a user enters a topic to the moment they receive the final report.

```
  User              CLI              Pipeline          Search         LLM Ensemble       Storage
   |                 |                  |                 |                |                |
   |  enter topic    |                  |                 |                |                |
   |---------------->|                  |                 |                |                |
   |                 |  run(topic)      |                 |                |                |
   |                 |----------------->|                 |                |                |
   |                 |                  |  expand query   |                |                |
   |                 |                  |---------------->|                |                |
   |                 |                  |  expanded       |                |                |
   |                 |                  |<----------------|                |                |
   |                 |                  |                 |                |                |
   |                 |                  |  fetch sources  |                |                |
   |                 |                  |---------------->|                |                |
   |                 |                  |  raw sources    |                |                |
   |                 |                  |<----------------|                |                |
   |                 |                  |                 |                |                |
   |                 |                  |  normalize      |                |                |
   |                 |                  |  deduplicate    |                |                |
   |                 |                  |  validate       |                |                |
   |                 |                  |                 |                |                |
   |                 |                  |  rank sources   |                |                |
   |                 |                  |--------------------------------->|                |
   |                 |                  |                 |  ensemble vote |                |
   |                 |                  |                 |  aggregate     |                |
   |                 |                  |  ranked results |                |                |
   |                 |                  |<---------------------------------|                |
   |                 |                  |                 |                |                |
   |                 |                  |  build path     |                |                |
   |                 |                  |                 |                |                |
   |                 |                  |  save results   |                |                |
   |                 |                  |--------------------------------------------->|
   |                 |                  |                 |                |  write JSON    |
   |                 |                  |                 |                |  write MD      |
   |                 |                  |                 |                |  write SQLite  |
   |                 |                  |  save confirmed |                |                |
   |                 |                  |<---------------------------------------------|
   |                 |                  |                 |                |                |
   |                 |  render report   |                 |                |                |
   |                 |<-----------------|                 |                |                |
   |  rich terminal  |                  |                 |                |                |
   |  report         |                  |                 |                |                |
   |<----------------|                  |                 |                |                |
```

---

## Key Design Decisions

### 1. Async Everywhere

Every network call, LLM invocation, and I/O operation is asynchronous. Fetching from seven platforms and calling three LLMs concurrently keeps total latency close to the slowest single call rather than the sum of all calls. The system uses Python's `asyncio` with `httpx` for HTTP and `asyncio.gather` for parallel execution.

### 2. Heuristic First, AI Second

The system always produces a result, even if every LLM provider is down. The heuristic scorer runs unconditionally and provides a baseline ranking. The AI consensus engine enhances this ranking when available but is never required. This design ensures the system is useful in any environment, including those with no LLM API access.

### 3. Consensus Over Single-Model

Multiple models voting on the same question produces more accurate rankings than any single model alone. The Mixture-of-Agents pattern reduces the impact of any one model's biases, hallucinations, or errors. When models disagree, a judge model breaks the tie with access to all reasoning.

### 4. Fail-Safe Stages

Every stage catches its own errors. A malformed response from one platform does not crash the fetcher. A failed LLM call does not crash the ranker. A broken source does not crash the validator. Errors are logged, collected in the state, and reported to the user at the end of the run.

### 5. Typed Models

Pydantic enforces the shape of data at every boundary between stages. If a stage produces output that does not match the expected schema, the error is caught immediately at the boundary rather than propagating silently and causing failures downstream.

### 6. No Hardcoded Secrets

All API keys and tokens are loaded from the `.env` file at startup. They are never hardcoded in source files, never logged in plain text, and never included in error messages. The logger masks any accidental inclusion of sensitive values.

---

## Reliability Mechanisms

The system includes multiple layers of protection against failures, rate limits, and data corruption.

| Mechanism              | Where It Applies                    | How It Works                                          |
|------------------------|-------------------------------------|-------------------------------------------------------|
| Rate limiting          | Per platform, per LLM provider      | Token bucket algorithm with configurable rates        |
| Retries                | All HTTP and LLM calls              | Exponential backoff with jitter, configurable max     |
| Caching                | API responses within a single run   | In-memory cache keyed by request URL and parameters   |
| Timeouts               | Every HTTP request and LLM call     | Configurable per-call timeout, default 30 seconds     |
| Fallbacks              | LLM ranking to heuristic ranking    | Automatic degradation when LLMs are unavailable       |
| Atomic file writes     | JSON, Markdown, and SQLite output   | Write to temp file, then rename to final path         |
| Error isolation        | Every pipeline stage                | Errors caught per-stage, logged, and collected        |
| Semaphore bounding     | Concurrent platform fetches         | Limits simultaneous requests to prevent API overload  |
| Guardrail validation   | LLM outputs before consumption      | Format, content, and integrity checks                 |
| Key masking            | Logging subsystem                   | Sensitive values redacted from all log output         |
```