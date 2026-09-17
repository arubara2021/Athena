# Future Plan

This document describes the future direction of the Research Agent project.

The current system can search multiple platforms, rank sources using heuristic and multi-model LLM consensus, generate learning paths, and save results to JSON, Markdown, and SQLite.

The next goal is to evolve it from a research search agent into a complete autonomous learning and knowledge system.

---

## Current State

The project currently supports:

- Multi-platform source search
- arXiv search
- Semantic Scholar search
- OpenAlex search
- GitHub search
- Wikipedia search
- Hugging Face search
- DuckDuckGo web search
- Source normalization
- Source deduplication
- Source validation
- Heuristic ranking
- Multi-model LLM consensus ranking
- Learning path generation
- JSON output
- Markdown output
- SQLite run history
- Rich terminal interface
- Interactive mode
- Unit tests
- Integration tests
- Installation scripts
- Documentation

---

## Immediate Stabilization Goals

Before building large future features, the following known issues should be closed.

### 1. Withdrawn paper filtering

Some arXiv results still include withdrawn papers.

Future improvement:

- Filter titles containing withdrawn phrases
- Filter arXiv metadata flags for withdrawn papers
- Filter retracted papers from Semantic Scholar and OpenAlex
- Add a source quality penalty for removed or unavailable papers

### 2. Run status persistence

The runs list may show old runs as RUNNING if the run was saved before final status was calculated.

Future improvement:

- Save SQLite state after final status is calculated
- Store final latency correctly
- Store completed_at timestamp
- Add run duration calculation
- Add partial success and failure reason columns

### 3. Better error visibility

Currently, individual platform failures may be hidden to protect the pipeline from crashing.

Future improvement:

- Collect warnings per platform
- Show platform warnings in the final report
- Add a --debug flag
- Add a warnings table in the CLI output
- Store warnings in SQLite

---

# Future Feature Phases

---

## Phase F1: Config Loader

The project currently has YAML configuration templates, but the runtime still uses environment variables and Python defaults.

The next step is to make the application read configuration from YAML files.

### Future capabilities

- Load configs/settings.yaml
- Load configs/sources.yaml
- Load configs/models.yaml
- Load configs/provider_config.yaml
- Load configs/ensemble.yaml
- Load configs/routing_rules.yaml
- Validate config files with Pydantic
- Override YAML values with environment variables
- Support development, staging, and production profiles
- Fail safely if required config is missing

### Tools to use

- PyYAML
- Pydantic
- pydantic-settings
- jsonschema optional

### Expected result

The system will become configurable without changing Python code.

---

## Phase F2: Prompt Loader

The project currently has prompt templates in prompts and llm/prompts.

The next step is to load prompts from files instead of keeping them only inside Python code.

### Future capabilities

- Load prompts from .txt files
- Support placeholders such as topic, goal, level, sources, and ranked_sources
- Validate required placeholders
- Version prompt templates
- Test prompts independently
- Switch prompt sets by profile
- Store prompt metadata

### Tools to use

- Python string.Template
- Jinja2 optional
- Pydantic prompt schema

### Expected result

Prompts become editable, versioned, and testable without touching core Python logic.

---

## Phase F3: Source Memory and RAG

This is one of the most important future upgrades.

The current system finds sources. The future system should remember them and allow chatting with them.

### Future capabilities

- Download paper PDFs
- Extract text from PDFs
- Split documents into chunks
- Generate embeddings
- Store chunks in a vector database
- Search saved sources semantically
- Ask questions against saved research
- Cite the exact source for every answer
- Build a personal research library

### Tools to use

- pypdf
- pdfplumber
- ChromaDB
- Qdrant
- LanceDB optional
- Gemini embedding models
- Mistral embedding models
- NVIDIA embedding models optional

### Expected result

The agent becomes NotebookLM-like.

The user will be able to say:

```text
Explain the top 5 papers from my Autonomous AI search.
Compare the consensus ranking reasons.
Create notes from the saved sources.
```

---

## Phase F4: Deep Research Agent

The current agent searches abstracts and metadata.

The future agent should perform deeper research.

### Future capabilities

- Read full papers
- Extract claims
- Extract methods
- Extract datasets
- Extract results
- Build evidence cards
- Compare multiple papers
- Detect contradictions
- Detect citation relationships
- Discover foundational papers
- Discover newer follow-up papers

### Tools to use

- Semantic Scholar citations API
- OpenAlex related works
- arXiv PDF download
- PDF parsing
- Vector search
- LLM summarization
- Consensus judging

### Expected result

The agent becomes a research analyst instead of only a search engine.

---

## Phase F5: Web Dashboard

The terminal interface is powerful, but a visual dashboard will make the system much easier to use.

### Future capabilities

- Search from a browser
- View ranked sources in a table
- Open learning path visually
- Filter by platform
- Filter by difficulty
- View saved runs
- Compare multiple runs
- Download Markdown and JSON
- Chat with saved sources
- View consensus voting results

### Tools to use

- FastAPI
- Next.js
- React
- Tailwind CSS
- WebSockets optional
- SQLite
- Vector database

### Expected result

The project becomes a full research application instead of only a CLI tool.

---

## Phase F6: Export and Integrations

The future system should connect with the user's existing learning and note-taking tools.

### Future integrations

- Export to Obsidian
- Export to Notion
- Export to Zotero
- Export BibTeX citations
- Export PDF summaries
- Export Anki flashcards
- Export weekly learning plan
- Export GitHub project README research section

### Tools to use

- Notion API
- Obsidian Markdown vault
- pyzotero optional
- bibtexparser
- Pandoc optional

### Expected result

Research output becomes usable inside the user's personal knowledge system.

---

## Phase F7: Scheduled Research

The future agent should monitor topics over time.

### Future capabilities

- Watch a topic daily or weekly
- Detect new papers
- Detect new repositories
- Detect trending topics
- Send email or terminal notifications
- Compare new results with old results
- Build a topic evolution timeline

### Tools to use

- APScheduler
- SQLite history comparison
- Email SMTP optional
- Telegram Bot API optional
- Discord webhook optional

### Expected result

The agent becomes an autonomous research watcher.

---

## Phase F8: Local Models

The current system depends on cloud LLM providers.

The future system should support local models for privacy and cost control.

### Future capabilities

- Run local LLMs
- Use local embedding models
- Switch between cloud and local providers
- Use local models for private notes
- Use cloud models for strong reasoning
- Mix local and cloud models in consensus ranking

### Tools to use

- Ollama
- LM Studio
- vLLM optional
- sentence-transformers
- local GGUF models

### Expected result

The agent can work partially or fully offline.

---

## Phase F9: Evaluation System

As the agent becomes more intelligent, it needs measurable quality checks.

### Future capabilities

- Evaluate ranking quality
- Evaluate learning path quality
- Evaluate summary accuracy
- Evaluate hallucination rate
- Compare model providers
- Compare prompt versions
- Store benchmark results

### Tools to use

- pytest
- custom evaluation datasets
- LLM-as-judge
- SQLite benchmark storage
- JSON reports

### Expected result

The system improves based on evidence instead of guesswork.

---

## Phase F10: Production Packaging

The final future goal is to make the project easy to install and deploy anywhere.

### Future capabilities

- Docker image
- docker-compose setup
- GitHub Actions CI
- automated tests
- lint checks
- type checks
- release builds
- PyPI package optional
- executable build optional

### Tools to use

- Docker
- GitHub Actions
- ruff
- pyright
- pytest
- pyproject.toml
- uv optional

### Expected result

The project becomes distributable and production-ready.

---

# Future Commands

Future CLI commands may include:

```text
python main.py ask "Explain the top papers from my last search"
python main.py watch "Autonomous AI" --interval daily
python main.py export --format notion
python main.py export --format obsidian
python main.py export --format bibtex
python main.py dashboard
python main.py compare run_id_1 run_id_2
python main.py library list
python main.py library chat
```

---

# Future Data Entities

The future system will need new data models.

## SourceDocument

Represents a downloaded source.

Fields:

- document_id
- source_id
- title
- url
- local_path
- file_type
- downloaded_at
- parse_status

## DocumentChunk

Represents a chunk of text from a source.

Fields:

- chunk_id
- document_id
- content
- chunk_index
- token_count
- embedding

## EvidenceCard

Represents a factual claim extracted from a source.

Fields:

- evidence_id
- source_id
- claim
- confidence
- quote
- page_number

## LearningNote

Represents a generated note.

Fields:

- note_id
- topic
- content
- source_ids
- created_at

## Conversation

Represents a chat session over saved sources.

Fields:

- conversation_id
- topic
- messages
- source_ids
- created_at

---

# Future Dependencies

Possible future dependencies:

| Feature | Dependency | Purpose |
|---|---|---|
| YAML config | PyYAML | Read YAML configuration |
| Prompt templates | Jinja2 | Advanced prompt rendering |
| PDF parsing | pypdf | Extract PDF text |
| PDF parsing | pdfplumber | Better PDF layout extraction |
| Vector database | ChromaDB | Local semantic search |
| Vector database | Qdrant | Scalable vector search |
| Scheduling | APScheduler | Watch topics over time |
| Web backend | FastAPI | Serve dashboard API |
| Web frontend | Next.js | Visual dashboard |
| Citations | bibtexparser | Export citations |
| Local models | Ollama | Local LLM support |
| Evaluation | pytest | Test and benchmark quality |

---

# Recommended Priority Order

The recommended order for future development is:

1. Fix withdrawn paper filtering completely
2. Fix SQLite run status and latency persistence
3. Build the YAML config loader
4. Build the prompt loader
5. Add PDF download and text extraction
6. Add embeddings and vector search
7. Build NotebookLM-style chat over saved sources
8. Build the web dashboard
9. Add export to Obsidian, Notion, and BibTeX
10. Add scheduled topic watching
11. Add local model support
12. Add evaluation benchmarks
13. Package with Docker and CI/CD

---

# Long-Term Vision

The final vision is:

```text
A personal autonomous research engine that finds, reads, remembers, ranks, explains, and tracks knowledge over time.
```

The system should eventually support:

```text
Search
Discovery
Ranking
Reading
Memory
Question answering
Note generation
Progress tracking
Long-term topic monitoring
```

This turns the project from a search tool into a complete autonomous learning companion.
