# Learning log

| Date | What I did | What I learned | Next |
|---|---|---|---|
| 2026-11-15 | Set up the AI-Journey workspace, cloned vLLM, set up Claude model routing (Opus, Haiku helpers, local Ollama) | Ollama is itself an inference engine (llama.cpp + Metal). vLLM is a serving engine built for many concurrent users on CUDA GPUs. PagedAttention = paging the KV cache in blocks. Continuous batching keeps the GPU busy | Run vLLM on a real GPU |
| 2026-11-17 | Ran `vllm_colab_intro.ipynb` on a free Colab Tesla T4 | T4 = Turing, 15 GiB VRAM, compute capability 7.5 (no bf16, no FlashAttention 2 → use fp16). GiB = 1024³ bytes. KV cache ≈ 28 KB/token for Qwen2.5-1.5B. Hit `ModuleNotFoundError: cutlass`: the module comes from `nvidia-cutlass-dsl`, not the `cutlass` pip package | Confirm the cutlass fix, finish the batching + server experiments |
