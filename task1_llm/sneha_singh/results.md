# Part 1 — GPT from scratch (Sneha Singh)

## Architecture
- Tokenizer: character-level (`char_to_idx` / `idx_to_char`)
- Split: 256 train / 64 val (own seed=670 split)
- n_layer=4, n_head=4, n_embd=256, block_size=128
- Causal mask: yes
- No prebuilt Transformer / attention modules

## Hyperparameters
- Epochs: 2 (smoke)
- Optimizer: AdamW, lr=0.001
- Warmup steps: 4 then cosine decay
- Batch size: 32
- Dropout: 0.1

## Metrics
See `metrics_report.csv` (filled by notebook section 1.5).
- Last train CE: 3.0619
- Last val CE: 3.0660
- Perplexity: 21.4551
- Top-1 next-char acc: 0.1949

## Hardware
- Device used: mps (Apple Silicon)
- Smoke run: True

## Notes
Why this config (for viva):
- 4 layers / 4 heads / 256 dim is small enough for TinyStories but still a real multi-block GPT.
- Character-level matches the lab (no BPE / tiktoken).
- Causal triu mask stops future leakage during training and generation.
