"""
Task 1 - Evaluation metrics.

All metrics required by the assignment's Task 1 metrics list live here:
  - training / validation cross-entropy loss
  - perplexity
  - bits-per-character (bpc)
  - generalization gap
  - top-1 next-character accuracy
  - distinct-1/2/3 (generation diversity)
  - repeated 4-gram rate
  - gradient norm stability summary (fed in from the training log)
  - parameter count
  - training/generation tokens-per-sec, peak memory, total training time
    (also fed in from the training log; helpers to format them live here too)
"""
import math
from collections import Counter
import numpy as np
import torch


def cross_entropy_to_perplexity(loss: float) -> float:
    return float(math.exp(loss))


def cross_entropy_to_bpc(loss: float) -> float:
    # loss is nats/char (torch cross_entropy default is natural log)
    return float(loss / math.log(2))


def generalization_gap(train_loss: float, val_loss: float) -> float:
    return float(val_loss - train_loss)


@torch.no_grad()
def top1_accuracy(model, X, Y, device, batch_size=256):
    model.eval()
    correct, total = 0, 0
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        yb = torch.from_numpy(Y[i : i + batch_size]).to(device)
        logits, _ = model(xb)
        preds = logits.argmax(dim=-1)
        correct += (preds == yb).sum().item()
        total += yb.numel()
    return correct / total


def _ngrams(seq, n):
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def distinct_n(texts, n):
    """Distinct-n over a list of generated strings: unique n-grams / total n-grams."""
    all_ngrams = []
    for t in texts:
        all_ngrams.extend(_ngrams(list(t), n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def repeated_4gram_rate(texts):
    """Fraction of 4-grams (across all generated text) that are repeats of an
    earlier occurrence within the same sample -- a simple, direct proxy for
    degenerate repetition loops."""
    total, repeated = 0, 0
    for t in texts:
        seen = set()
        grams = _ngrams(list(t), 4)
        total += len(grams)
        for g in grams:
            if g in seen:
                repeated += 1
            else:
                seen.add(g)
    if total == 0:
        return 0.0
    return repeated / total


def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def format_metrics_report(d: dict) -> str:
    lines = ["# Task 1 Metrics Report\n"]
    for k, v in d.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.6f}")
        else:
            lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)
