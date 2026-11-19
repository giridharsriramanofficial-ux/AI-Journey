"""Train your GPT on tiny-Shakespeare (character level).

    uv run python train.py                 # tiny: ~0.8M params, a few minutes, sanity check
    uv run python train.py --size small    # ~10.7M params (Karpathy's config), ~30-60 min on MPS

Writes out/<size>/ckpt.pt, out/<size>/loss.png and prints a sample at the end.
"""
import argparse
import csv
import math
import time
from pathlib import Path

import torch

from gpt import GPT, GPTConfig

HERE = Path(__file__).parent
SIZES = {
    "tiny": dict(block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.0),
    "small": dict(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2),
}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"  # Apple-silicon GPU
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=SIZES, default="tiny")
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    args = ap.parse_args()
    max_iters = args.max_iters or (2000 if args.size == "tiny" else 5000)
    device = pick_device()
    torch.manual_seed(1337)

    # ---- data: characters -> integer ids ------------------------------------------
    path = HERE / "data" / "shakespeare.txt"
    if not path.exists():
        raise SystemExit("run `uv run python prepare.py` first")
    text = path.read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}  # val = text the model never trains on

    cfg = GPTConfig(vocab_size=len(chars), **SIZES[args.size])

    def get_batch(split: str):
        d = splits[split]
        ix = torch.randint(len(d) - cfg.block_size - 1, (args.batch_size,))
        x = torch.stack([d[i:i + cfg.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + cfg.block_size] for i in ix])  # targets = inputs shifted by 1
        return x.to(device), y.to(device)

    # ---- model + optimizer -------------------------------------------------------
    model = GPT(cfg).to(device)
    print(f"device={device}  params={model.num_params() / 1e6:.2f}M  iters={max_iters}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    warmup = max_iters // 20

    def lr_at(it: int) -> float:
        # Linear warmup, then cosine decay to 10% of peak. Standard for LLM pretraining.
        if it < warmup:
            return args.lr * (it + 1) / warmup
        progress = (it - warmup) / max(1, max_iters - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    @torch.no_grad()
    def estimate_loss(iters: int = 50) -> dict[str, float]:
        model.eval()
        out = {}
        for split in splits:
            losses = [model(*get_batch(split))[1].item() for _ in range(iters)]
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    # ---- training loop -----------------------------------------------------------
    out_dir = HERE / "out" / args.size
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    t0 = time.time()
    for it in range(max_iters + 1):
        if it % args.eval_every == 0 or it == max_iters:
            l = estimate_loss()
            history.append((it, l["train"], l["val"]))
            print(f"iter {it:5d}  train {l['train']:.3f}  val {l['val']:.3f}  "
                  f"lr {lr_at(it):.2e}  {time.time() - t0:.0f}s")
        if it == max_iters:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoid exploding updates
        opt.step()

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "chars": chars}, out_dir / "ckpt.pt")
    with open(out_dir / "losses.csv", "w", newline="") as f:
        csv.writer(f).writerows([("iter", "train", "val"), *history])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        its, tr, va = zip(*history)
        plt.plot(its, tr, label="train")
        plt.plot(its, va, label="val")
        plt.xlabel("iteration"); plt.ylabel("cross-entropy loss"); plt.legend()
        plt.savefig(out_dir / "loss.png", dpi=120)
    except ImportError:
        pass

    # ---- sample ------------------------------------------------------------------
    model.eval()
    start = torch.tensor([[stoi["\n"]]], device=device)
    sample = model.generate(start, max_new_tokens=500, temperature=0.8, top_k=40)
    print("\n--- sample ---\n" + "".join(itos[i] for i in sample[0].tolist()))
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
