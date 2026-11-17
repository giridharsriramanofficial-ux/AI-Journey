# vLLM dev setup on this Mac (Apple silicon, CPU backend)

`vllm/` is a clone of upstream. To contribute you'll need your own fork:

```bash
# 0. one-time tools
curl -LsSf https://astral.sh/uv/install.sh | sh       # uv (system python is 3.9, too old)
xcode-select --install                                 # Apple Clang, if missing

# 1. fork and point remotes (needs `gh auth login`)
cd ~/Documents/AI-Journey/03-vllm-contrib/vllm
gh repo fork --remote --remote-name origin             # upstream becomes 'upstream'

# 2. Python 3.12 env + CPU build (macOS auto-sets VLLM_TARGET_DEVICE=cpu)
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
uv pip install -r requirements/cpu.txt
uv pip install -e .                                    # compiles C++ CPU ops, takes a while

# 3. contributor tooling
uv pip install "pre-commit>=4.5.1" pytest pytest-asyncio
pre-commit install

# 4. smoke test: serve a tiny model, then hit the OpenAI-compatible API
vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --max-model-len 2048
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"hi"}]}'
```

Optional, for GPU inference via Metal/MLX: https://github.com/vllm-project/vllm-metal

## Per-PR workflow
```bash
git fetch upstream && git switch -c fix/<short-name> upstream/main
# ...change + add/adjust tests...
pytest tests/<relevant_test>.py -x        # many GPU tests will skip/fail on Mac; that's expected
pre-commit run -a
git commit -s -m "[Bugfix] <what>"        # -s = DCO sign-off (required)
git push origin HEAD && gh pr create --repo vllm-project/vllm
```

## Learning path through the code (read in this order)
1. `vllm/entrypoints/openai/`: HTTP request → engine
2. `vllm/v1/engine/`: engine core loop
3. `vllm/v1/core/sched/`: scheduler (continuous batching)
4. `vllm/v1/core/kv_cache_manager.py`: PagedAttention block management
5. `vllm/model_executor/models/llama.py`: compare with your nanoGPT
6. `vllm/model_executor/layers/quantization/`: how quantized weights run

Ask Claude to explain each one: the `scout` subagent finds the code, Opus explains it.
