"""
Task 1.3 - Training and text generation.

- Cross-entropy loss (built into GPTScratch.forward)
- Linear warmup -> cosine decay LR schedule
- Minimum 10 epochs (default 12)
- Tracks: grad norm per step (stability / NaN detection), tokens/sec,
  peak memory (RSS), total wall-clock training time
- Saves: checkpoint, raw unedited training log (JSONL, one line per step --
  do not hand-edit this file, it is the evidence trail for grading),
  train/val loss curve plot
"""
import argparse
import json
import math
import os
import resource
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import GPTScratch
from metrics import (
    cross_entropy_to_perplexity, cross_entropy_to_bpc, generalization_gap,
    top1_accuracy, param_count, format_metrics_report,
)


def lr_lambda(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    # cosine decay to 10% of peak LR
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, X, Y, device, batch_size=256):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        yb = torch.from_numpy(Y[i : i + batch_size]).to(device)
        _, loss = model(xb, yb)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../data_processed")
    ap.add_argument("--out_dir", default="../outputs")
    ap.add_argument("--ckpt_dir", default="../checkpoints")
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=6638)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=None,
                     help="optional hard cap on total optimizer steps, for CPU-time-budgeted runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    vocab_size, block_size = meta["vocab_size"], meta["block_size"]

    Xtr = np.load(os.path.join(args.data_dir, "X_train.npy"))
    Ytr = np.load(os.path.join(args.data_dir, "Y_train.npy"))
    Xval = np.load(os.path.join(args.data_dir, "X_val.npy"))
    Yval = np.load(os.path.join(args.data_dir, "Y_val.npy"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = GPTScratch(vocab_size, block_size, args.n_layer, args.n_head, args.n_embd, args.dropout).to(device)
    n_params = param_count(model)
    print(f"Parameter count: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(Xtr) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_lambda(s, warmup_steps, total_steps)
    )

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log_f = open(log_path, "w")  # raw, unedited log -- evidence trail, do not hand-edit after the run

    train_losses_per_epoch, val_losses_per_epoch = [], []
    grad_norms, nan_events = [], 0
    tokens_seen = 0
    t_start = time.time()
    step = 0

    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(Xtr))
        epoch_loss, n_batches = 0.0, 0
        t_epoch = time.time()

        for bstart in range(0, len(Xtr), args.batch_size):
            if args.max_steps and step >= args.max_steps:
                break
            idx = perm[bstart : bstart + args.batch_size]
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(Ytr[idx]).to(device)

            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
            is_nan = not math.isfinite(loss.item()) or not math.isfinite(grad_norm)
            if is_nan:
                nan_events += 1
            grad_norms.append(grad_norm)

            optimizer.step()
            scheduler.step()

            tokens_seen += xb.numel()
            epoch_loss += loss.item()
            n_batches += 1
            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                rec = {
                    "step": step, "epoch": epoch, "loss": loss.item(),
                    "grad_norm": grad_norm, "lr": scheduler.get_last_lr()[0],
                    "tokens_per_sec": tokens_seen / max(elapsed, 1e-8),
                    "elapsed_sec": elapsed, "nan_or_inf": is_nan,
                }
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()

        if args.max_steps and step >= args.max_steps:
            train_losses_per_epoch.append(epoch_loss / max(n_batches, 1))
            val_loss = evaluate(model, Xval, Yval, device)
            val_losses_per_epoch.append(val_loss)
            print(f"[epoch {epoch}] (stopped early at max_steps) train_loss={train_losses_per_epoch[-1]:.4f} val_loss={val_loss:.4f}")
            break

        train_loss = epoch_loss / n_batches
        val_loss = evaluate(model, Xval, Yval, device)
        train_losses_per_epoch.append(train_loss)
        val_losses_per_epoch.append(val_loss)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"time={time.time()-t_epoch:.1f}s")

        torch.save(
            {"model_state": model.state_dict(), "config": vars(args), "meta": meta, "epoch": epoch},
            os.path.join(args.ckpt_dir, f"ckpt_epoch{epoch}.pt"),
        )

    total_train_time = time.time() - t_start
    log_f.close()

    torch.save(
        {"model_state": model.state_dict(), "config": vars(args), "meta": meta},
        os.path.join(args.ckpt_dir, "ckpt_final.pt"),
    )

    # ---- loss curve plot ----
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses_per_epoch, label="train loss", marker="o")
    plt.plot(val_losses_per_epoch, label="val loss", marker="o")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss (nats/char)")
    plt.title("Task 1 - Training / Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # ---- final metrics ----
    peak_mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    final_train_loss = train_losses_per_epoch[-1]
    final_val_loss = val_losses_per_epoch[-1]
    acc = top1_accuracy(model, Xval, Yval, device)

    metrics = {
        "device": str(device),
        "parameter_count": n_params,
        "epochs_trained": len(train_losses_per_epoch),
        "final_train_cross_entropy": final_train_loss,
        "final_val_cross_entropy": final_val_loss,
        "train_perplexity": cross_entropy_to_perplexity(final_train_loss),
        "val_perplexity": cross_entropy_to_perplexity(final_val_loss),
        "train_bits_per_char": cross_entropy_to_bpc(final_train_loss),
        "val_bits_per_char": cross_entropy_to_bpc(final_val_loss),
        "generalization_gap": generalization_gap(final_train_loss, final_val_loss),
        "val_top1_next_char_accuracy": acc,
        "mean_grad_norm": float(np.mean(grad_norms)),
        "max_grad_norm": float(np.max(grad_norms)),
        "nan_or_inf_events": nan_events,
        "total_training_time_sec": total_train_time,
        "avg_train_tokens_per_sec": tokens_seen / total_train_time,
        "peak_memory_mb": peak_mem_kb / 1024.0,
        "train_losses_per_epoch": train_losses_per_epoch,
        "val_losses_per_epoch": val_losses_per_epoch,
    }
    with open(os.path.join(args.out_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out_dir, "metrics_report.md"), "w") as f:
        f.write(format_metrics_report({k: v for k, v in metrics.items() if not isinstance(v, list)}))

    print("Training complete.")
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()
