# ATHENA — All Commands

Every command you can run. Organized by purpose. Run from the project root.

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Verification after setup

```powershell
python main.py --help
python main.py version
python main.py doctor
```

---

## Core commands

### Version and system info

```powershell
python main.py version
```

### Environment check

```powershell
python main.py doctor
```

### Show configuration

```powershell
python main.py config
```

### List registered tools

```powershell
python main.py tools
```

### Aggregate statistics over saved runs

```powershell
python main.py stats
python main.py stats --limit 50
python main.py stats --limit 500
```

---

## Saved runs

### List saved runs

```powershell
python main.py runs list
python main.py runs list --limit 10
python main.py runs list --limit 100
```

### Show one run in detail

```powershell
python main.py runs show <request-id>
```

---

## Search

### Basic search

```powershell
python main.py search "dense retrieval methods"
python main.py search "quantum computing"
python main.py search "organic chemistry basics"
python main.py search "machine learning"
python main.py search "photosynthesis"
```

### Search with maximum sources

```powershell
python main.py search "dense retrieval methods" --max 20 --display 20
python main.py search "machine learning" --max 30 --display 30
python main.py search "physics" --max 50 --display 50
```

### Search with a learning goal

```powershell
python main.py search "dense retrieval" --goal "Learn from basics to advanced"
python main.py search "python" --goal "Master advanced patterns" --level advanced
```

### Search with a target level

```powershell
python main.py search "chemistry" --level beginner
python main.py search "physics" --level intermediate
python main.py search "machine learning" --level advanced
python main.py search "javascript" --level "beginner to advanced"
```

### Search across specific platforms

```powershell
python main.py search "dense retrieval" --platform arxiv
python main.py search "chemistry" --platform wikipedia --platform wikibooks
python main.py search "python" --platform github --platform huggingface
```

### Search with source type filter

```powershell
python main.py search "chemistry" --source-type book
python main.py search "programming" --source-type repository
python main.py search "physics" --source-type course --source-type video
```

### Search with mode

```powershell
python main.py search "dense retrieval" --mode fast
python main.py search "dense retrieval" --mode balanced
python main.py search "dense retrieval" --mode deep
```

### Search with LLM options

```powershell
python main.py search "dense retrieval" --llm-expansion
python main.py search "dense retrieval" --no-llm-ranking
python main.py search "dense retrieval" --llm-path
python main.py search "dense retrieval" --no-llm-expansion --no-llm-ranking
```

### Search with RAG indexing

```powershell
python main.py search "dense retrieval" --max 20 --rag
python main.py search "quantum computing" --max 20 --rag
```

### Search with output options

```powershell
python main.py search "dense retrieval" --json
python main.py search "dense retrieval" --markdown
python main.py search "dense retrieval" --quiet
python main.py search "dense retrieval" --open
```

### Search with display limit

```powershell
python main.py search "dense retrieval" --display 5
python main.py search "dense retrieval" --display 30
```

### Combined example

```powershell
python main.py search "dense retrieval methods" --max 20 --display 20 --level "beginner to advanced" --mode balanced --rag
```

---

## Ask

Convenience wrapper around `search` with defaults from `configs/settings.yaml`.

```powershell
python main.py ask "dense retrieval"
python main.py ask "quantum computing"
python main.py ask "chemistry"
python main.py ask "machine learning" --mode fast
python main.py ask "physics" --json
python main.py ask "javascript" --quiet
```

---

## Agent

### Basic agent run

```powershell
python main.py agent "learn quantum computing"
python main.py agent "learn pathology basics"
python main.py agent "learn dense retrieval from basics"
```

### Agent with iteration and budget controls

```powershell
python main.py agent "learn quantum computing" --iterations 3 --budget 8000
python main.py agent "learn chemistry" --iterations 5 --budget 15000
python main.py agent "learn machine learning" --iterations 15 --budget 40000
```

### Agent with source limits

```powershell
python main.py agent "learn quantum computing" --max 20
python main.py agent "learn quantum computing" --max 30 --display 30
```

### Agent with topic override

```powershell
python main.py agent "study retrieval" --topic "dense retrieval"
```

### Agent with level

```powershell
python main.py agent "learn quantum computing" --level beginner
python main.py agent "learn machine learning" --level advanced
```

### Agent with mode

```powershell
python main.py agent "learn quantum computing" --mode fast
python main.py agent "learn quantum computing" --mode balanced
python main.py agent "learn quantum computing" --mode deep
```

### Agent with output options

```powershell
python main.py agent "learn quantum computing" --json
python main.py agent "learn quantum computing" --quiet
```

### Combined agent example

```powershell
python main.py agent "learn dense retrieval from basics" --iterations 5 --budget 15000 --max 20 --level beginner --mode balanced
```

---

## Chat

### Single question

```powershell
python main.py chat "what is in the corpus"
python main.py chat "what topics are indexed"
python main.py chat "explain hybrid dense-sparse retrieval" --no-stream
python main.py chat "what is retrieval-augmented generation" --no-stream
```

### Chat with more passages

```powershell
python main.py chat "explain dense retrieval" --top-k 20 --no-stream
python main.py chat "explain dense retrieval" --top-k 10 --no-stream
```

### Chat with diversity control

```powershell
python main.py chat "explain dense retrieval" --diversity 0 --no-stream
python main.py chat "explain dense retrieval" --diversity 3 --no-stream
```

### Chat across all runs

```powershell
python main.py chat "explain dense retrieval" --all-runs --no-stream
```

### Chat filtered to one run

```powershell
python main.py chat "explain dense retrieval" --run-id <request-id> --no-stream
```

### Chat with rebuild

```powershell
python main.py chat "explain dense retrieval" --rebuild-index --no-stream
```

### Chat with diagnostics

```powershell
python main.py chat "explain dense retrieval" --explain --no-stream
```

### Chat with JSON output

```powershell
python main.py chat "what is in the corpus" --json
```

### Chat without streaming

```powershell
python main.py chat "explain dense retrieval" --no-stream
```

### Chat with session id

```powershell
python main.py chat "explain dense retrieval" --session-id my-session --no-stream
```

### Combined chat example

```powershell
python main.py chat "explain dense retrieval" --top-k 20 --diversity 0 --all-runs --explain --no-stream
```

---

## Interactive chat

### Open the REPL

```powershell
python main.py chat --interactive
```

### Open with custom settings

```powershell
python main.py chat --interactive --top-k 20 --diversity 0
```

### Open with a specific session

```powershell
python main.py chat --interactive --session-id my-session
```

### Inside the REPL

```
/help
/new
/reset
/clear
/history
/sessions
/save sessions/my-session.json
/rebuild
/explain
/quit
/exit
```

---

## Global options

Place these before the command name.

```powershell
python main.py --theme paper version
python main.py --theme mono version
python main.py --theme midnight version
python main.py --no-color version
python main.py --verbose search "dense retrieval"
python main.py --debug search "dense retrieval"
```

---

## Compile checks

```powershell
python -m py_compile .\main.py
python -m py_compile .\cli\app.py
python -m py_compile .\cli\chat.py
python -m py_compile .\cli\search.py
python -m py_compile .\cli\agent.py
python -m py_compile .\cli\render.py
python -m py_compile .\cli\progress.py
python -m py_compile .\storage\html_writer.py
python -m py_compile .\storage\markdown_writer.py
python -m py_compile .\agent\controller.py
python -m py_compile .\core\pipeline.py
python -m py_compile .\ranking\scorer.py
```

Or compile everything in one line:

```powershell
Get-ChildItem -Recurse -Filter *.py | Where-Object { $_.FullName -notmatch '\\__pycache__\\' } | ForEach-Object { python -m py_compile $_.FullName }
```

---

## Open output artifacts

### Open the latest HTML report

```powershell
Start-Process (Get-ChildItem .\output\reports\*.html | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
```

### Open the latest Markdown report

```powershell
Start-Process (Get-ChildItem .\output\markdown\*.md | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
```

### Open the latest JSON output

```powershell
Start-Process (Get-ChildItem .\output\json\*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
```

### List every output file

```powershell
Get-ChildItem .\output -Recurse -File | Select-Object FullName, Length, LastWriteTime
```

---

## Clean outputs

```powershell
Remove-Item .\output\json\*     -Force -ErrorAction SilentlyContinue
Remove-Item .\output\markdown\* -Force -ErrorAction SilentlyContinue
Remove-Item .\output\reports\*  -Force -ErrorAction SilentlyContinue
```

---

## Clean caches

```powershell
Get-ChildItem -Recurse -Filter __pycache__ -Directory | Remove-Item -Recurse -Force
Remove-Item .\.pytest_cache -Recurse -Force -ErrorAction SilentlyContinue
```

---

## Logs

### Count trace lines

```powershell
(Get-Content .\logs\debug_trace.jsonl).Count
```

### Search trace for ranking events

```powershell
Get-Content .\logs\debug_trace.jsonl -Tail 500 | Select-String -Pattern "ranking_relevance_gate_proportional|ranking_topic_gate_applied|llm_ranking_success|ranking_quality_floor"
```

### Search trace for LLM calls

```powershell
Get-Content .\logs\debug_trace.jsonl -Tail 500 | Select-String -Pattern "llm_call_success|llm_call_failed"
```

### Search trace for errors

```powershell
Get-Content .\logs\debug_trace.jsonl -Tail 500 | Select-String -Pattern "failed|error"
```

### Tail the application log

```powershell
Get-Content .\logs\research_agent.log -Tail 50 -Wait
```

Press `Ctrl+C` to stop following.

---

## Git — first time

```powershell
git check-ignore -v .env
git init
git branch -M main
git config user.name "Your Name"
git config user.email "you@example.com"
git add .
git status --short
git commit -m "Initial commit: ATHENA autonomous research agent"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

---

## Git — day to day

```powershell
git status
git add .
git commit -m "describe your change"
git push
```

### Pull latest

```powershell
git pull
```

### See commit history

```powershell
git log --oneline -20
```

### Undo a commit that has not been pushed

```powershell
git reset --soft HEAD~1
```

### Undo a change to one file

```powershell
git checkout -- path/to/file.py
```

### Remove a file from tracking without deleting it

```powershell
git rm --cached path/to/file
```

---

## Verification suite

Run these in order. They prove the system is healthy end to end.

```powershell
python main.py version
python main.py doctor
python main.py config
python main.py tools
python main.py stats --limit 10
python main.py runs list --limit 5
python main.py search "dense retrieval methods" --max 20 --display 20
python main.py search "dense retrieval" --max 20 --display 20 --rag
python main.py agent "learn quantum computing" --iterations 3 --budget 8000
python main.py chat "what topics are indexed" --top-k 20 --no-stream
python main.py chat "explain hybrid dense-sparse retrieval" --top-k 20 --no-stream
```

---

## Quick reference card

```
LIGHT COMMANDS                       HEAVY COMMANDS
─────────────────                    ─────────────────
version                              search
doctor                               ask
config                               agent
tools                                chat
stats                                chat --interactive
runs list
runs show
```

Every command in the left column finishes in under a second. The right column loads the pipeline, RAG engine, or agent, and takes between 20 seconds and a minute depending on mode and network.