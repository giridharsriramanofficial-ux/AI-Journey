#!/usr/bin/env bash
# Ask a local Ollama model (clean text output, no spinners). Usage:
#   ask-local.sh [model] "prompt"
#   cat file.py | ask-local.sh qwen2.5-coder:7b "Add docstrings to this code"
# Default model: code-coach (your custom qwen2.5-coder-based tutor).
set -euo pipefail
if [[ $# -ge 2 ]]; then model="$1"; shift; else model="code-coach"; fi
prompt="$*"
if [[ ! -t 0 ]]; then
  prompt="$prompt"$'\n\n```\n'"$(cat)"$'\n```'
fi
python3 - "$model" "$prompt" <<'PY'
import json, sys, urllib.request
model, prompt = sys.argv[1], sys.argv[2]
req = urllib.request.Request(
    "http://127.0.0.1:11434/api/generate",
    data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=300) as r:
        print(json.load(r)["response"])
except OSError as e:
    sys.exit(f"Ollama not reachable ({e}). Start it with: ollama serve  (or open the Ollama app)")
PY
