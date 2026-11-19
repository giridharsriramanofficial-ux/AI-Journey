"""A tiny decoder-only transformer (GPT). You write the parts marked TODO.

Work order: MLP -> CausalSelfAttention -> Block -> GPT.forward.
After each one, run `uv run pytest -q` and watch more tests pass.

References:
- Karpathy, "Let's build GPT: from scratch, in code, spelled out" (YouTube)
- Vaswani et al., "Attention Is All You Need", section 3.2

Shape names used everywhere: B = batch, T = time (sequence length),
C = channels (n_embd), nh = number of heads, hs = head size (C // nh).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65    # 65 distinct characters in tiny-Shakespeare
    block_size: int = 128   # max context length in tokens
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


class MLP(nn.Module):
    """Position-wise feed-forward: C -> 4C -> GELU -> C."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: fc -> GELU -> proj -> dropout. (B, T, C) in, (B, T, C) out.
        raise NotImplementedError("TODO: MLP.forward")


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t may only look at positions <= t."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must divide evenly into heads"
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # q, k, v in one matmul
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular True/False mask: mask[t, s] is True when s <= t.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # TODO:
        # 1. q, k, v = self.qkv(x).split(C, dim=2)            each (B, T, C)
        # 2. reshape each to (B, nh, T, hs):  .view(B, T, nh, hs).transpose(1, 2)
        # 3. att = (q @ k.transpose(-2, -1)) / sqrt(hs)       (B, nh, T, T)
        #    Why divide by sqrt(hs)? Without it, softmax saturates as hs grows.
        # 4. att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        # 5. att = softmax(att, dim=-1), then self.attn_dropout
        # 6. y = att @ v -> (B, nh, T, hs) -> transpose(1, 2).contiguous().view(B, T, C)
        # 7. return self.resid_dropout(self.proj(y))
        # Write it by hand first. Later, compare with F.scaled_dot_product_attention(is_causal=True).
        raise NotImplementedError("TODO: CausalSelfAttention.forward")


class Block(nn.Module):
    """One transformer layer: pre-norm attention and MLP, each wrapped in a residual."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: x = x + attn(ln1(x)); x = x + mlp(ln2(x)); return x
        # The residual "+" is what lets gradients flow through deep stacks.
        raise NotImplementedError("TODO: Block.forward")


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # what the token is
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # where it is
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        # Small init -> near-uniform predictions at step 0 -> loss starts near ln(vocab_size).
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids. Returns (logits (B, T, vocab), loss or None)."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        # TODO:
        # 1. pos = torch.arange(T, device=idx.device)
        # 2. x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))     (B, T, C)
        # 3. run x through every block, then self.ln_f
        # 4. logits = self.lm_head(x)                                  (B, T, vocab)
        # 5. if targets is given: loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        #    else loss = None
        raise NotImplementedError("TODO: GPT.forward")

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None) -> torch.Tensor:
        """Autoregressive sampling: predict a token, append it, repeat."""
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]          # crop to the context window
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature      # only the last position matters
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
