"""Checks for your GPT. Run: uv run pytest -q"""
import math

import pytest
import torch

from gpt import GPT, MLP, Block, CausalSelfAttention, GPTConfig

CFG = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=4, n_embd=64)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_mlp_shape():
    x = torch.randn(2, 10, CFG.n_embd)
    assert MLP(CFG)(x).shape == x.shape


def test_attention_shape():
    x = torch.randn(2, 10, CFG.n_embd)
    assert CausalSelfAttention(CFG)(x).shape == x.shape


def test_attention_is_causal():
    """Changing the token at position t must not change outputs at positions < t."""
    attn = CausalSelfAttention(CFG).eval()
    x = torch.randn(1, 12, CFG.n_embd)
    y1 = attn(x)
    x2 = x.clone()
    x2[:, 7] += 1.0
    y2 = attn(x2)
    assert torch.allclose(y1[:, :7], y2[:, :7], atol=1e-6), "positions < 7 saw the future"
    assert not torch.allclose(y1[:, 7:], y2[:, 7:]), "positions >= 7 should be affected"


def test_attention_matches_pytorch_reference():
    attn = CausalSelfAttention(CFG).eval()
    x = torch.randn(2, 16, CFG.n_embd)
    B, T, C = x.shape
    hs = C // CFG.n_head
    q, k, v = attn.qkv(x).split(C, dim=2)
    q, k, v = (t.view(B, T, CFG.n_head, hs).transpose(1, 2) for t in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = attn.proj(ref.transpose(1, 2).contiguous().view(B, T, C))
    assert torch.allclose(attn(x), ref, atol=1e-5)


def test_block_shape():
    x = torch.randn(2, 10, CFG.n_embd)
    assert Block(CFG)(x).shape == x.shape


def test_gpt_initial_loss_is_near_uniform():
    """At init the model should be ~equally unsure about every token: loss ≈ ln(vocab)."""
    model = GPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (4, CFG.block_size))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, CFG.block_size, CFG.vocab_size)
    assert abs(loss.item() - math.log(CFG.vocab_size)) < 0.3


def test_gpt_no_targets_returns_none_loss():
    logits, loss = GPT(CFG)(torch.zeros(1, 5, dtype=torch.long))
    assert logits.shape == (1, 5, CFG.vocab_size) and loss is None


def test_gpt_can_overfit_one_batch():
    """A correct model memorizes a single small batch. If this fails, something is broken."""
    model = GPT(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, CFG.vocab_size, (4, CFG.block_size))
    y = torch.roll(x, -1, dims=1)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 1.0, f"loss stuck at {loss.item():.2f}"


def test_generate_appends_tokens():
    model = GPT(CFG).eval()
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=40)
    assert out.shape == (1, 41)
