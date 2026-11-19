# 02-build-and-train — build your own small "Claude/ChatGPT"

Goal: go through every stage a frontier chat model goes through, at a scale that fits on an
M5 MacBook (16 GB, no CUDA). The result is not a Claude competitor. It is a full understanding of
how one is made, plus a working chat model you trained and served yourself.

## The pipeline, frontier vs. you

| Stage | What the big labs do | What you do here |
|---|---|---|
| 1. Pretraining | ~1T-param model, trillions of tokens, thousands of GPUs | 10–50M-param GPT from scratch on Shakespeare / TinyStories |
| 2. Instruction tuning (SFT) | Human-written example conversations | LoRA fine-tune Qwen2.5-0.5B/1.5B on a chat dataset |
| 3. Preference tuning (RLHF / DPO) | Human rankings + reinforcement learning | DPO on `ultrafeedback_binarized` |
| 4. Evaluation | Big benchmark suites, red-teaming | `lm-evaluation-harness` + your own 30-question set |
| 5. Serving | GPU clusters | Ollama / vLLM (CPU build) + a chat UI |

## Phase 1 — GPT from scratch (2–3 weeks) → `../01-foundations/gpt/`
- Watch Karpathy's "Let's build GPT" and "Let's build the GPT Tokenizer".
- Fill in the TODOs in `01-foundations/gpt/gpt.py` until `pytest` passes.
- Train on character-level Shakespeare, then on TinyStories (~20M params, a few hours).
- **Done when:** you have a loss curve, samples, and can explain every line.

## Phase 2 — Pretraining for real (2 weeks)
- Read nanoGPT, then Karpathy's **nanochat** (tokenizer → pretrain → SFT → RL → eval → web UI).
- Learn: LR warmup + cosine decay, gradient accumulation, mixed precision, checkpointing,
  scaling laws (Chinchilla: ~20 tokens per parameter).
- Write your own BPE tokenizer and swap it in for the character-level one.
- *Optional:* rent 8×H100 for ~4 h (~$100) and run nanochat's speedrun to get a ~500M chat
  model trained from scratch. Everything else in this plan is free.

## Phase 3 — Make an assistant by fine-tuning (2–3 weeks)
- Base model: `Qwen2.5-1.5B` (4-bit). Use 0.5B for fast iteration. 3B is tight on 16 GB.
- Tooling: `mlx-lm` (LoRA on Apple silicon); `trl` for DPO.
- Steps: base → SFT (subset of `HuggingFaceH4/ultrachat_200k`, or your own data) → DPO.
- Compare base vs. SFT vs. SFT+DPO on the same prompts. This is where you *see* alignment work.
- **Done when:** the model is on Hugging Face with a model card.

## Phase 4 — Evaluate and serve (1–2 weeks)
- `lm-evaluation-harness` on HellaSwag, ARC, GSM8K, plus your own question set.
  Use Claude as a judge only alongside your own judgment.
- Merge LoRA → convert to GGUF → run in Ollama, and/or serve via the vLLM CPU build
  (ties into `03-vllm-contrib`: KV cache, scheduler and sampling become concrete).
- Chat UI: Open WebUI, or a Gradio app you write.

## Phase 5 (optional) — Specialize
- A domain model, e.g. a code-review assistant for your stack.
- RAG on top (see `04-ai-for-swe`).
- Distillation: generate training data with a stronger model (check its license terms first).

## Budget
- Hardware: the Mac covers Phases 1, 3 and 4. A cloud GPU (RunPod / Lambda / Colab, ~$1–3/h)
  only helps in Phase 2.
- Stack: PyTorch (MPS), MLX / mlx-lm, HF `transformers` / `datasets` / `trl`, Ollama, `lm-eval`.
- Time: ~2–3 months at 8–10 h/week.

## Realistic expectations
A fine-tuned 1.5B model will chat and follow instructions, but it will hallucinate confidently
and be weak at hard reasoning. That's normal for the size, not a failure.

## Progress
- [ ] Phase 1 — GPT from scratch
- [ ] Phase 2 — nanoGPT / nanochat, own tokenizer
- [ ] Phase 3 — SFT + DPO on Qwen2.5
- [ ] Phase 4 — eval + serving
- [ ] Phase 5 — specialization
