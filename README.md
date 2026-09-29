# ATHENA

**Autonomous Research Agent · Curriculum Synthesis · Local Corpus Intelligence**

A single-machine research system that takes a natural-language learning goal and returns a fully assembled curriculum. It queries diverse open platforms in parallel, ranks what it finds through a heuristic-plus-consensus pipeline, and produces a learning path with real sources attached to every step. Every run is saved as JSON, Markdown, HTML, and a shared SQLite row.

The output is not a chat reply. It is a document you can open, print, and study from.

---

## What makes it different

**Deterministic pipeline under an autonomous controller.** The lower layer — query understanding, fetching, normalization, deduplication, heuristic scoring, consensus ranking, summarization, path building, persistence — is a fully testable pipeline with no hidden state. The upper layer is an agent loop that plans, invokes pipeline stages as tools, reflects on the output, and re-plans. The controller absorbs the unpredictability of live APIs. The pipeline stays predictable.

**Multi-signal relevance with a heuristic floor.** Before any language model sees a source, it receives a composite score over eight dimensions: token overlap with the query, platform authority, source-type quality, log-saturated citation impact, recency, content completeness, artifact availability, and author completeness. This is the reason a run still produces a ranked list when every LLM provider is offline.

**Consensus ranking with a judge fallback.** The top candidates are dispatched concurrently to independent models. Each ranking is fingerprinted and voted on. If agreement reaches a configurable threshold, the majority ranking wins. If not, a judge model reads every vote and reasons about the conflict. If every provider fails, the heuristic order is returned unchanged.

**Two-gate retrieval that never starves the model.** A relevance gate keeps sources whose overlap with the concept exceeds a proportional floor. A topic-presence gate keeps sources whose tokens share meaningful matches with the query. Both gates include a rescue path: if the gates would drop below the minimum working set, the highest-scoring dropped sources return to the pool.

**Local vector store with hybrid retrieval.** Every ranked source is embedded and indexed into a SQLite database with two schemas — a parent-document table and a chunk table — indexed by FTS5 for keyword search and by packed floating-point blobs for semantic search. Chat retrieval combines both through a weighted hybrid, and it decides between chunk and parent search based on the number of content-bearing tokens in the question.

**Catalog mode.** When the question is about the corpus itself — what it contains, how many sources, what topics, what platforms — retrieval is bypassed entirely. A snapshot of the vector store is passed to a language model under a catalog-specific prompt that says: describe the corpus using only this data.

**Multi-key rotation.** Provider keys accept comma-separated lists. Every provider that exposes multiple credentials is rotated transparently. When a key hits a rate limit, the next one is tried.

---

## Quick start

```
python main.py version
python main.py doctor
python main.py search "dense retrieval methods" --max 20 --display 20
python main.py chat "what topics are indexed" --top-k 20 --no-stream
python main.py agent "learn quantum computing" --iterations 3 --budget 8000
```

---

## Commands

### Diagnostics

| Command | Purpose |
| :--- | :--- |
| `python main.py version` | Show Athena version, Python version, executable path, platform |
| `python main.py doctor` | Check environment: folders, providers, database, vector store, tool registry |
| `python main.py config` | Show active providers, models, fallback chains, directories |
| `python main.py tools` | List every registered agent tool with category, description, and token cost |
| `python main.py stats --limit 100` | Aggregate statistics over recent runs |

### Runs

| Command | Purpose |
| :--- | :--- |
| `python main.py runs list --limit 20` | List saved runs from the SQLite store |
| `python main.py runs show <request-id>` | Show one run with full metadata and sources |

### Search

The core pipeline. Runs query understanding, concurrent fetching, deduplication, ranking, summarization, path building, and persistence.

```
python main.py search "dense retrieval methods" --max 20 --display 20
```

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--max` | 20 | Maximum final sources to keep |
| `--display` | 10 | Rows shown in the final report |
| `--goal` | Learn from basics to advanced | Learning goal |
| `--level` | beginner to advanced | Target learner level |
| `--platform` | all enabled | Filter to one or more platforms |
| `--source-type` | all types | Filter to one or more source types |
| `--llm-expansion` | off | Enable LLM-based query expansion |
| `--llm-ranking` | on | Enable consensus ranking |
| `--llm-path` | off | Enable LLM learning path generation |
| `--rag` | off | Index results into the vector store for chat |
| `--mode` | balanced | Pipeline mode: `fast`, `balanced`, `deep` |
| `--json` | off | Emit raw JSON |
| `--markdown` | off | Emit Markdown |
| `--quiet` | off | Minimal output |
| `--open` | off | Open the saved Markdown report |

### Ask

A convenience wrapper around `search` with the CLI defaults from `configs/settings.yaml`.

```
python main.py ask "dense retrieval methods"
```

### Agent

The autonomous controller. Plans, iterates a Think–Act–Observe–Reflect loop, allocates a token budget across categories, and synthesizes from accumulated material.

```
python main.py agent "learn quantum computing" --iterations 3 --budget 8000
```

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--iterations` | 15 | Maximum loop iterations |
| `--budget` | 40000 | Token budget |
| `--max` | 20 | Maximum final sources |
| `--topic` | goal | Topic override |
| `--level` | beginner to advanced | Target learner level |
| `--display` | 10 | Rows shown in the report |
| `--mode` | balanced | Agent mode: `fast`, `balanced`, `deep` |
| `--json` | off | Emit raw JSON |
| `--quiet` | off | Minimal output |

### Chat

Ask questions over the indexed corpus. Retrieval is hybrid: keyword FTS5 combined with semantic embedding search, with an optional LLM relevance rerank.

```
python main.py chat "explain hybrid dense-sparse retrieval" --top-k 20 --no-stream
```

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--top-k` | 5 | Number of passages to retrieve |
| `--diversity` | 0 | Max sources per platform (0 = unlimited) |
| `--run-id` | latest | Limit chat to one saved run |
| `--all-runs` | off | Search across all saved runs |
| `--rebuild-index` | off | Rebuild the vector index before querying |
| `--explain` | off | Print full retrieval diagnostics |
| `--stream` | on | Stream the answer |
| `--session-id` | new | Continue a saved chat session |
| `--json` | off | Emit raw JSON |

### Interactive chat

```
python main.py chat --interactive --top-k 20 --diversity 0
```

Inside the REPL:

| Command | Action |
| :--- | :--- |
| `/help` | Show the full command list |
| `/new`, `/reset` | Start a fresh session |
| `/clear` | Clear the current session history |
| `/history` | Show turns in this session |
| `/sessions` | List sessions on disk |
| `/save PATH` | Export the session to a file |
| `/rebuild` | Rebuild the vector index from output JSON |
| `/explain` | Toggle diagnostics on every answer |
| `/quit`, `/exit` | Leave the chat |

### Global options

These can be placed before the command name.

```
python main.py --theme paper version
python main.py --theme mono search "dense retrieval"
python main.py --no-color version
python main.py --verbose search "dense retrieval"
python main.py --debug search "dense retrieval"
```

| Option | Purpose |
| :--- | :--- |
| `--theme` | Theme: `midnight`, `paper`, `mono` |
| `--no-color` | Disable all color output |
| `--verbose`, `-v` | Verbose logging |
| `--debug` | Show tracebacks on error |

---

## How a run works

```
┌──────────────────────────────────────────────────────────────┐
│                    QUERY UNDERSTANDING                       │
│  correct topic · extract concept · extract keywords          │
│  detect intent · detect level · detect subject domains       │
└───────────────────────────┬──────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                        FETCHING                              │
│  concurrent across selected platforms under a semaphore      │
│  per-platform rate limits · retries with backoff · caching   │
└───────────────────────────┬──────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                    NORMALIZE + DEDUPE                        │
│  DOI match · arXiv match · URL fingerprint · fuzzy title     │
└───────────────────────────┬──────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                   HEURISTIC SCORING                          │
│  eight weighted dimensions + learner-alignment multiplier    │
└───────────────────────────┬──────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                     CONSENSUS RANK                           │
│  concurrent models → fingerprint → majority → judge          │
└───────────────────────────┬──────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                       SUMMARIZE                              │
│  per-source summaries with difficulty + key topics           │
└───────────────────────────┬──────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                     LEARNING PATH                            │
│  ordered steps · objectives · resources · estimated time     │
└───────────────────────────┬──────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                      PERSISTENCE                             │
│  JSON · Markdown · HTML · SQLite row · optional vector store │
└──────────────────────────────────────────────────────────────┘
```

Every stage is bounded by a token budget. Every external failure is absorbed and logged as an event. No single platform, model, or file write can abort the run.

---

## Output artifacts

Every run produces four artifacts inside `output/`:

| Artifact | Location | Contents |
| :--- | :--- | :--- |
| JSON | `output/json/` | Full payload: query, sources, ranked list, summaries, learning path, errors, warnings, saved files |
| Markdown | `output/markdown/` | Human-readable report with aligned tables and linked sources |
| HTML | `output/reports/` | Self-contained styled report with theme toggle, score bars, timeline learning path, print support |
| SQLite | `data/research_agent.db` | Shared store: runs, sources, learning steps, memory, episodes |

The HTML report is fully self-contained. No external fonts, scripts, or stylesheets. Open it directly in any browser. Toggle dark and light. Print it and the layout adapts.

---

## Configuration

All configuration lives in `configs/`.

| File | Purpose |
| :--- | :--- |
| `settings.yaml` | Modes, budgets, search defaults, RAG settings, CLI defaults |
| `models.yaml` | Model roles, fallback chains, provider assignments |
| `routing_rules.yaml` | Per-task routing: which model chain serves which stage |
| `ranking.yaml` | Scoring weights, gate thresholds, alignment settings |
| `domains.yaml` | Subject-domain aliases and platform mappings |
| `sources.yaml` | Platform enablement, rate limits, source types, domain tags |
| `token_budgets.yaml` | Per-category token budgets and circuit breaker thresholds |
| `agent_config.yaml` | Autonomous agent defaults |

---

## Requirements

- Python 3.10 or newer
- An internet connection for source fetching
- At least one LLM provider API key in `.env` for LLM-backed stages

Key dependencies: `httpx`, `pydantic`, `pydantic-settings`, `typer`, `rich`, `PyYAML`, `python-dotenv`, `feedparser`.

The heuristic scorer, deduplication, normalization, and persistence stages run without any LLM provider. Search continues to produce a ranked list when every provider is offline.

---

## Known behaviour

**Topic gates are token-based, not semantic.** On queries with a single strong concept word, the topic gate can admit sources that share the token but not the meaning. The LLM ranker is designed to catch those on the deep mode; on balanced it trusts the gate. Improving the gate to a semantic check is the next step.

**The funnel shows shrinkage.** If twelve sources are retrieved and seven ranked, the report explains that the relevance and topic-presence gates dropped five. This is the intended behaviour, not data loss.

**Chat refuses when the corpus does not contain the answer.** The generator is instructed to say so explicitly rather than invent content. When a question is not covered by the indexed material, the answer will say that. It will not hallucinate.

---

## License

This project is provided as-is for research and educational use.

---

**ATHENA** · Autonomous Research Agent · Built on deterministic retrieval with an autonomous controller above it.