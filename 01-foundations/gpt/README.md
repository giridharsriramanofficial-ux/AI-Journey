# Phase 1 — GPT from scratch

Part of the plan in `../../02-build-and-train/README.md`.

## Setup (once)
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv (Python env manager)
cd ~/Documents/AI-Journey/01-foundations/gpt
uv sync                                           # Python 3.12 venv + torch, numpy, matplotlib, pytest
uv run python prepare.py                          # downloads tiny-Shakespeare to data/
```

## Loop
1. Fill one TODO in `gpt.py`, in this order: `MLP` → `CausalSelfAttention` → `Block` → `GPT.forward`.
2. `uv run pytest -q`. Every test fails at first, and more pass as you fill in each piece.
3. When all tests pass: `uv run python train.py` (tiny, minutes) → `uv run python train.py --size small`.

## What "good" looks like
- Loss at step 0 ≈ ln(65) ≈ 4.17, which means the model is guessing uniformly over the characters.
- Tiny: val loss ~1.7–1.9 after 2000 iters. Output looks like Shakespeare, with made-up words.
- Small: val loss ~1.5. Character names, line breaks, mostly real words.
- If val loss rises while train loss keeps falling, the model is overfitting. That's what dropout is for.

## Questions to answer in `notes/LOG.md` when done
- Why divide attention scores by √(head size)?
- What breaks if you remove the causal mask? (Try it: train loss collapses, but generation is garbage. Why?)
- Why pre-norm (LayerNorm before attention) instead of after?
- How many parameters are in the embeddings vs. the blocks for the `small` config?

## Next
Train on TinyStories with a BPE tokenizer, then go to Phase 2 (nanoGPT / nanochat).
