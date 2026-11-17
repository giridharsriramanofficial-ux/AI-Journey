# AI-Journey workspace — instructions for Claude

Goal of the owner: become an AI/ML engineer / data scientist, contribute to open-source AI
(starting with vLLM), deeply understand how models are built/trained/served, and use AI to be a
better software engineer. Treat every session as **teaching + doing**: explain the "why", not just the fix.

## Model routing (who does what)

The main session runs on **Claude Opus 5.5** and owns all *thinking*:
architecture understanding, picking issues, designing fixes, reviewing diffs, debugging,
explaining ML concepts, and the final say on anything that gets committed or posted.

Delegate cheap, well-scoped work — and always verify the result before using it:

| Task | Delegate to |
|---|---|
| Sweep the vLLM repo to locate code ("where is X defined / who calls Y") | `scout` subagent (Haiku) |
| Summarize long files, logs, test output, issue threads, papers | `summarizer` subagent (Haiku) |
| Boilerplate: docstrings, type hints, simple unit-test skeletons, commit-message drafts, regex, shell one-liners, flashcards/quiz questions | `local-coder` subagent → Ollama `qwen2.5-coder:7b` (free, offline) |
| Quick "explain this snippet" for the learner | `scripts/ask-local.sh` (Ollama `code-coach`) |

Rules:
- Never delegate: choosing *what* to change, correctness reasoning, kernel/scheduler/KV-cache logic, final review, anything posted to GitHub.
- Never send private info (email, tokens) to any model or service.
- Local models are weak at large-context reasoning — give them small, self-contained inputs (< ~200 lines).

## vLLM contribution rules (from docs.vllm.ai/en/stable/contributing)
- Python 3.12 venv via `uv`. On this Mac (Apple silicon, no CUDA) build the **CPU** backend: see `03-vllm-contrib/SETUP.md`.
- `pre-commit install` in the repo; run `pre-commit run -a` before pushing.
- Every commit needs DCO: `git commit -s`.
- PR titles use prefixes: `[Bugfix]`, `[Doc]`, `[Test]`, `[Misc]`, `[Frontend]`, `[CPU]`, `[Model]` ...
- Many GPU tests can't run locally — say so in the PR and rely on CI.
- Commit or push only when the owner asks. Comment on an issue before starting work on it.

## Layout
- `01-foundations/` — math, PyTorch, transformer from scratch
- `02-build-and-train/` — train/fine-tune small models locally (MLX / PyTorch MPS)
- `03-vllm-contrib/` — vLLM clone (`vllm/`), setup, issue tracker, notes
- `04-ai-for-swe/` — using LLMs in day-to-day engineering (RAG, agents, eval)
- `notes/` — learning log (`LOG.md`)
