# AI-Journey

From software engineer to AI/ML engineer: learn how models work, build and train them, contribute to
vLLM, and use AI in day-to-day engineering.

## Is vLLM a good place to start?

**Yes, as a second step, not the first.** vLLM is one of the best open-source projects for learning
how LLMs actually *run*: attention, the KV cache (PagedAttention), batching/scheduling, quantization,
model architectures, and OpenAI-compatible serving. It's widely used in industry, very active, and
merged PRs there look strong on a resume.

Some caveats, based on the repo as of Nov 2026:
- **The "good first issue" list isn't really for beginners.** Of 14 open ones, 11 are already
  assigned. Most are deep refactors (MoE oracles, KV-cache update extraction, torch.compile
  collectives) with 10 to 100 comments of history. Easier entry points:
  docs fixes, missing tests, CPU-backend bugs, small `[Frontend]`/OpenAI-API bugs, and
  issues labelled `documentation`, `misc`, or `bug` with a clear repro. Track candidates in
  `03-vllm-contrib/ISSUES.md`.
- **Hardware:** this Mac (M5, 16 GB, no NVIDIA GPU) can build vLLM's **CPU backend** (FP32/FP16) and use
  `vllm-metal` for GPU inference via MLX. Most CUDA kernels and GPU tests can't run here, so pick
  Python-level work: frontend, API, config, docs, tests, CPU/Apple paths.
- vLLM teaches *inference*. It doesn't teach how to *train* models, so phases 1 and 2 below cover that.

## Roadmap (about 6 months, 8 to 10 hrs/week; adjust as you go)

| Phase | Weeks | Folder | Outcome |
|---|---|---|---|
| 1. Foundations | 1–6 | `01-foundations` | Python/NumPy fluency, linear algebra & probability refresh, PyTorch basics, **build a GPT from scratch** (Karpathy "Zero to Hero" + nanoGPT) |
| 2. Build & train | 5–12 | `02-build-and-train` | Train a tiny LM on Mac (PyTorch MPS / MLX), fine-tune a small model with LoRA (MLX-LM or HF PEFT), evaluate it, classic ML with scikit-learn + one Kaggle comp (data-science side) |
| 3. Serve & contribute | 6 → ongoing | `03-vllm-contrib` | Build vLLM (CPU), serve Qwen2.5-0.5B, read the engine code path, land 1 doc/test PR, then 1 bugfix PR |
| 4. AI for SWE | parallel | `04-ai-for-swe` | RAG over your own codebase, a tool-using agent, LLM evals; use at your job |

Phases overlap on purpose: start reading vLLM code (phase 3) once you've built the GPT in phase 1.
Code you wrote yourself makes vLLM's `model_executor/models/llama.py` much easier to follow.

### Core resources
- Karpathy, *Neural Networks: Zero to Hero* (YouTube) + `nanoGPT` / `nanochat`
- *Dive into Deep Learning* (d2l.ai), free; *Hands-On ML* (Géron) for the data-science side
- Hugging Face LLM course; MLX & `mlx-lm` examples (best for training on Apple silicon)
- Papers: *Attention Is All You Need*, *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM), *LoRA*, *FlashAttention*
- vLLM docs → "Design" section (architecture, paged attention, scheduler)

## How Claude is set up here
See `CLAUDE.md`. Opus does the reasoning, Haiku subagents (`scout`, `summarizer`) do cheap repo
sweeps and summaries, and the `local-coder` subagent and `scripts/ask-local.sh` send boilerplate
to your Ollama models (`qwen2.5-coder:7b`, `code-coach`) at no cost.
Start Claude Code from this folder so it picks these up: `cd ~/Documents/AI-Journey && claude`.

Log progress in `notes/LOG.md`.
