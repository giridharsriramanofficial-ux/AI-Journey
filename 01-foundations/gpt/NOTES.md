# Study notes — how `gpt.py` works

My notes from building a GPT from scratch (Phase 1). Interactive version of the big picture:
open `explainers/meet-think-repeat.html` in a browser.

---

## 1. The big picture

A GPT reads text and predicts the **next character**, at every position at once.

```
"To be or" → token ids [32, 53, 1, ...]       (one number per character)
   → tok_emb + pos_emb                         (each id → vector of 128 numbers: what it is + where it is)
   → Block × 4                                 (attention: tokens look at earlier tokens; MLP: each token "thinks")
   → ln_f → lm_head                            (each vector → 65 scores, one per possible next character)
   → cross_entropy(scores, true next chars)    (the loss: how wrong the predictions were)
```

Training = nudge the weights so the loss goes down. That's all pretraining is, at any scale.
Claude / ChatGPT follow the same overall recipe, much bigger, plus many refinements.

**Shape names:** B = batch, T = time (sequence length), C = channels (`n_embd`),
nh = number of heads, hs = head size (C // nh).

---

## 2. `GPTConfig` — the settings panel

Six numbers decide the model's size. The code never changes; only these do.
`@dataclass` auto-generates the constructor, so `GPTConfig(n_layer=6)` works.

| Setting | Meaning | Why this value | If you raise it |
|---|---|---|---|
| `vocab_size = 65` | Size of the "alphabet" | Set by the data: tiny-Shakespeare has 65 distinct characters | (Not a choice.) Real models use ~50k word pieces (BPE) |
| `block_size = 128` | Context window: max tokens seen at once (~2–3 lines) | Attention cost grows with T²: 128² = 16k scores vs 1024² = 1M | Longer patterns, but slower and more memory |
| `n_embd = 128` | Width: numbers per token | Params grow with C² | 2× width ≈ 4× params |
| `n_layer = 4` | Depth: number of Blocks | Cost grows linearly | More rounds of reasoning |
| `n_head = 4` | Parallel attention "searches" | Head size 128/4 = 32 (typical: 32–64) | **No extra params** — same 128 numbers, cut thinner |
| `dropout = 0.0` | Fraction randomly zeroed in training | Tiny model *underfits*, so dropout only hurts | `small` uses 0.2: 10.7M params can memorize 1M chars |

Rule of thumb for dropout: small model or lots of data → little or none; big model and little data → more.

### Same code, different scale

| | `tiny` | `small` | GPT-2 small | GPT-3 |
|---|---|---|---|---|
| vocab_size | 65 | 65 | 50,257 | 50,257 |
| block_size | 128 | 256 | 1,024 | 2,048 |
| n_layer | 4 | 6 | 12 | 96 |
| n_head | 4 | 6 | 12 | 96 |
| n_embd | 128 | 384 | 768 | 12,288 |
| Parameters | 0.8M | 10.7M | 124M | 175B |

---

## 3. Parameter count: where "12 × C²" comes from

`nn.Linear(a, b)` has `a × b` weights (every input connects to every output) + `b` biases.

| Layer in a Block | Shape | Weights |
|---|---|---|
| Attention `qkv` | C → 3C | 3C² |
| Attention `proj` | C → C | 1C² |
| MLP `fc` | C → 4C | 4C² |
| MLP `proj` | 4C → C | 4C² |
| **Total** | | **12C²** |

- Attention = 4C² (⅓), MLP = 8C² (⅔) of each block.
- "About" because biases (9C) and two LayerNorms (4C) add 13C — under 1%.
- C = 128: 12 × 16,384 = 196,608 + 1,664 = **198,272** exactly.
- Doubling C quadruples params. GPT-3: 12 × 12,288² ≈ 1.8B per block × 96 ≈ 175B.

Check: `uv run python -c "from gpt import Block, GPTConfig; print(sum(p.numel() for p in Block(GPTConfig()).parameters()))"`

Whole tiny model: embeddings ~25k + blocks 4 × ~198k + lm_head ~8k ≈ **0.82M**.

---

## 4. The GPT class

### `__init__` — the parts list
- **`tok_emb`** (65 × 128): a lookup table. Row 46 = the vector for `'h'`. An id is just a label;
  the embedding swaps it for a vector with meaning. Similar characters end up with similar vectors.
- **`pos_emb`** (128 × 128): one vector per *seat* 0–127. Attention alone ignores order
  ("dog bites man" = "man bites dog"), so position is added in. Only 128 seats → `T <= block_size`.
- **`blocks`**: `nn.ModuleList`, not a plain list — a plain list hides the weights from the optimizer.
- **`ln_f`**: final LayerNorm. **`lm_head`** (128 → 65): turns each vector into 65 scores (*logits*).

### `_init_weights` — a fair start
Weights ~ Normal(0, 0.02), biases = 0. Small weights → all logits ≈ 0 → every character equally likely
→ starting loss ≈ **ln(65) ≈ 4.17**. Random (not zero) so neurons don't all learn the same thing.

### `forward` — the pipeline (example: `"hell"` → should predict `"o"`)
```python
pos = torch.arange(T, device=idx.device)              # [0,1,2,3]; same chip as the data
x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # what + where; (T,C) broadcasts over B
for block in self.blocks:
    x = block(x)
x = self.ln_f(x)
logits = self.lm_head(x)                               # (B, T, 65)
loss = None
if targets is not None:
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
return logits, loss
```

**Every position predicts:** `h`→?, `he`→?, `hel`→?, `hell`→`o`. One 128-char sequence gives
128 training examples at once. The causal mask keeps this honest.

**Targets** = input shifted by one: input `h e l l` → target `e l l o`.

**Cross-entropy:** take the probability given to the correct answer, loss = −ln(p).

| p(correct) | loss |
|---|---|
| 1/65 (guess) | 4.17 |
| 0.5 | 0.69 |
| 0.9 | 0.11 |
| 1.0 | 0 |

`.view(-1, V)` flattens (B, T, 65) → (B·T, 65) because `cross_entropy` wants a flat list.

### `generate` — how it writes (exactly how chatbots do)
One token at a time, in a loop: predict → pick → append → repeat.

| Code | Meaning |
|---|---|
| `@torch.no_grad()` | Not training → skip gradient bookkeeping |
| `idx[:, -block_size:]` | Only the last 128 tokens fit (why chatbots "forget" long chats) |
| `logits[:, -1, :]` | Only the last position predicts the next token |
| `/ temperature` | <1 = safer, >1 = more random (train.py uses 0.8) |
| `top_k` | Keep the 40 likeliest, rule out absurd picks |
| `torch.multinomial` | Roll weighted dice → same prompt, different answers |

---

## 5. The Block — "meet, think, add notes"

One transformer layer. Stack 4 (GPT-3: 96). Each block **adds notes** to a shared running document
called the **residual stream**; it never rewrites it.

```python
x = x + self.attn(self.ln1(x))   # meeting: gather info from earlier tokens
x = x + self.mlp(self.ln2(x))    # thinking: each token processes it alone
return x
```

### LayerNorm — tidy up the numbers
Per token: shift its numbers to mean 0, scale to spread ~1, then apply a learned scale and offset.
`[2, 4, 6, 8]` → `[-1.34, -0.45, 0.45, 1.34]`. Keeps numbers from drifting huge/tiny → stable training.

### The residual `+`
1. **Nothing is lost:** a block with nothing to add outputs ~0, and `x + 0 = x`.
2. **Learning reaches every layer:** without it the gradient signal fades through many layers
   (vanishing gradients). The `+` is an express lane straight back → 100-layer models are trainable.

### Pre-norm
LayerNorm is applied to the *copy* going into attention/MLP, never to the main stream. Trains more
stably than the 2017 paper's post-norm.

### Meet → think, visually (`"hell"`)
```
          'h'          'e'          'l'          'l'
 MEETING  [h]◄─────────[e]◄─────────[l]◄─────────[l]     each token hears itself
 (attn)    every token hears from itself + everyone to its LEFT
             ▼            ▼            ▼            ▼
 THINKING [MLP]        [MLP]        [MLP]        [MLP]   same MLP, same weights,
 (mlp)                                                    each token alone, in parallel
```

Who can hear whom (the causal mask):
```
                  'h'   'e'   'l'   'l'
 token 0 'h'       ✅    ❌    ❌    ❌
 token 1 'e'       ✅    ✅    ❌    ❌
 token 2 'l'       ✅    ✅    ✅    ❌
 token 3 'l'       ✅    ✅    ✅    ✅   ← sees everything → best prediction
```

Across blocks (roughly — the model learns its own division of labor):
Block 1 letter pairs → Block 2 parts of words → Block 3 words → Block 4 what fits here.

---

## 6. Attention — the meeting

Each token's vector only knows itself. Attention lets it collect information from earlier tokens.

### Query, Key, Value
| | Meaning | Library analogy |
|---|---|---|
| **q** | "What am I looking for?" | Your search request |
| **k** | "What do I contain?" | The label on the book's spine |
| **v** | "What I'll hand over if picked" | The book's content |

### Where the weights are
```
 ① qkv   each token ALONE: writes its q, k, v                [3C² weights]
 ⇄       compare q with earlier k's → softmax → mix their v's
         ★ the ONLY step where tokens exchange info          [no weights]
 ② proj  each token ALONE: sums up what it heard, merges heads [1C² weights]
```
⇄ has no learned weights: **who listens to whom is decided by the content**, not fixed wiring.
That's why the same model links "sat" to "cat" in one sentence and to "dog" in another.

### Worked example: "the cat sat" (2 numbers per vector)
`sat` wants to know *who sat*.

| Token | q | k | v |
|---|---|---|---|
| the | — | `[0, 1]` filler word | `[0, 1]` |
| cat | — | `[2, 0]` a thing that acts | `[5, 0]` animal, furry… |
| sat | `[2, 0]` "who's doing this?" | `[0.5, 0.5]` an action | `[1, 2]` |

```
a) score  q·k:     the 0      cat 4      sat 1
b) scale  ÷√2:     the 0.00   cat 2.83   sat 0.71
c) mask:          nothing after "sat" → nothing masked (a later token would get −∞)
d) softmax:        e^x = 1.00, 16.9, 2.03 (sum 19.95) → the 5%, cat 85%, sat 10%
e) mix the v's:    0.05·[0,1] + 0.85·[5,0] + 0.10·[1,2] = [4.35, 0.25]   ← mostly cat's info
f) proj:           learned matrix → e.g. [2.18, 1.56]   (merges 4 heads, shared format)
g) residual:       x_sat = x_sat + [2.18, 1.56]
```

### The 7 steps in code
```python
B, T, C = x.shape
nh, hs = self.n_head, C // self.n_head
q, k, v = self.qkv(x).split(C, dim=2)                    # each (B, T, C)
q = q.view(B, T, nh, hs).transpose(1, 2)                  # (B, nh, T, hs) — same for k, v
k = k.view(B, T, nh, hs).transpose(1, 2)
v = v.view(B, T, nh, hs).transpose(1, 2)
att = (q @ k.transpose(-2, -1)) / hs ** 0.5               # (B, nh, T, T) scores
att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))  # no peeking ahead
att = F.softmax(att, dim=-1)                               # rows sum to 1; e^−∞ = 0
att = self.attn_dropout(att)
y = att @ v                                                # mix values (B, nh, T, hs)
y = y.transpose(1, 2).contiguous().view(B, T, C)           # glue heads back
return self.resid_dropout(self.proj(y))
```

- **Why heads?** 4 smaller attentions in parallel, each looking for a different relationship.
  `.transpose(1, 2)` puts heads next to batch so `@` runs all heads at once (it multiplies the last two dims).
- **Why ÷√hs?** A dot product of 32 random numbers lands around ±6 → softmax goes ~100% on one
  token → learning stalls. Dividing brings scores back to ~±1.
- **Why `.contiguous()`?** `transpose` only changes how memory is read; `.view` needs it laid out in order.
- **The mask** (lower triangle) is what makes it a *causal* left-to-right language model.

---

## 7. MLP — the thinking

Position-wise: the same small network on every token separately.
Research suggests much of a model's stored "knowledge" lives here.

```python
x = self.fc(x)        # (B, T, C)  -> (B, T, 4C)   expand: 512 "feature detectors"
x = F.gelu(x)         # non-linearity: keep positives, squash negatives
x = self.proj(x)      # (B, T, 4C) -> (B, T, C)    shrink back so x + mlp(x) works
x = self.dropout(x)
return x
```

- **Why 4×?** Room to work; convention from the original Transformer paper.
- **Why GELU is essential:** two linear layers in a row = one linear layer (matrix × matrix = matrix).
  The bend in the middle is what lets the network learn non-straight-line patterns.
- **Dropout** randomly zeroes values in training to reduce memorizing; off in `model.eval()`.
- Params: (128×512 + 512) + (512×128 + 128) = **131,712** ≈ ⅔ of each block.

---

## 8. Where to run what

| Option | Use for |
|---|---|
| Mac (M5, 16 GB, MPS) | Writing/debugging, tests, Phase 1 training, MLX LoRA ≤1.5B, Ollama |
| Kaggle (~30 free GPU-h/week) | Long runs (Phase 2 pretraining, bigger fine-tunes) |
| Colab (free T4) | Quick CUDA experiments, vLLM on GPU |
| GitHub | Version control + portfolio — doesn't train models |

Rule: develop locally; train remotely only when a run would take > ~1 hour on the Mac.

---

## 9. Questions to be able to answer (interview prep)
- Why divide attention scores by √(head size)?
- What breaks without the causal mask? (Train loss collapses, generation is garbage — why?)
- Why pre-norm instead of post-norm?
- Why does the residual connection make deep models trainable?
- Why can't you remove GELU from the MLP?
- Do more heads add parameters? Why not?
- Where do most of a transformer's parameters live, and how do they scale with width?
- Why is the starting loss ≈ ln(vocab_size)?
- How do temperature and top-k change generation?
