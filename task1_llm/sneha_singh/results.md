# Part 1 — GPT from scratch (Sneha Singh)

## Architecture
- Tokenizer: character-level (`char_to_idx` / `idx_to_char`)
- Split: 100000 train / 10000 val (own seed=670 split)
- n_layer=4, n_head=4, n_embd=256, block_size=128
- Causal mask: yes
- No prebuilt Transformer / attention modules

## Hyperparameters
- Epochs: 10 (full)
- Optimizer: AdamW, lr=0.001
- Warmup steps: 100 then cosine decay
- Batch size: 32
- Dropout: 0.1

## Metrics
See `metrics_report.csv` (filled by notebook section 1.5).
- Last train CE: 0.9120
- Last val CE: 0.8571
- Perplexity: 2.3562
- Top-1 next-char acc: 0.7288

## Hardware
- Device used: cuda: NVIDIA GeForce RTX 5090
- Smoke run: False

## Notes
Why this config (for viva):
- 4 layers / 4 heads / 256 dim is small enough for TinyStories but still a real multi-block GPT.
- Character-level matches the lab (no BPE / tiktoken).
- Causal triu mask stops future leakage during training and generation.
