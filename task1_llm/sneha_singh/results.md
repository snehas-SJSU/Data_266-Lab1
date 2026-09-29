# Part 1 — GPT from scratch (Sneha Singh)

I trained a small character-level GPT on TinyStories. The attention code is mine. I did not use `nn.Transformer` or `nn.MultiheadAttention`.

## What I built
- Character tokenizer (`char_to_idx` / `idx_to_char`), not BPE
- 4 layers, 4 heads, embedding size 256, context of 128 characters
- A causal mask, so a position cannot see future characters
- My own split from `TinyStories-train.txt`: 100,000 train windows and 10,000 val windows, seed 670

## Training
This is the full run (`smoke` is false), not the old 2-epoch laptop smoke test.

- 10 epochs
- AdamW, learning rate 0.001
- 100 warmup steps, then cosine decay
- Batch size 32, dropout 0.1

## How it did
- Train cross-entropy 0.912, validation cross-entropy 0.857
- Perplexity 2.36
- Next-character accuracy 0.729
- No NaN losses. About 3.25 million parameters. Training took 457 seconds.

The loss curve is `outputs/loss_curves.png`. A short generation is `outputs/samples.txt`. Every column is in `metrics_report.csv`.

## Hardware
NVIDIA GeForce RTX 5090, CUDA, mixed precision. Peak memory was about 349 MB. The raw log is `reproducibility/raw_logs/sneha_singh/task1_llm/train_full.log`.

`TinyStories-train.txt` is not in the repo. The file is 1.79 GB, over GitHub’s 100 MB limit: https://drive.google.com/drive/folders/12MFVOo6T3QRW3X6THiDkh5svtNgL13W_?usp=share_link

## Why this size
Four layers is small, but it is still a real stack of blocks, and it stays inside the “write it yourself” rule. Character tokens are what the lab asked for. The causal mask is what stops the model from peeking ahead while it trains and while it writes.
