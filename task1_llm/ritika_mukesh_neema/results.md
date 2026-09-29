# Task 1 — Results: Character-Level GPT on TinyStories

**Author:** Ritika Mukesh Neema
**Personal seed:** 6638 (derived from student ID 019306638)

## Architecture
GPT built entirely from scratch (no nn.Transformer/nn.MultiheadAttention): 4 layers, 4 heads, 128 embedding dim, manual causal multi-head self-attention, manual LayerNorm, GELU feed-forward, learnable token + positional embeddings, pre-norm transformer blocks. 882,688 parameters.

## Data
Full TinyStoriesV2-GPT4 corpus: 1,824,825,663 train characters, 22,493,387 validation characters, vocab_size=221 (built from train text only). 100,000 fixed-length training sequences and 10,000 validation sequences (block_size=256), built with seed=6638 controlling the sequence-sampling positions.

## How to reproduce
Run from the src/ directory:
1. python data.py --train_txt <path> --val_txt <path> --seed 6638
2. python train.py --seed 6638
3. python generate.py

## Training configuration
12 epochs, batch_size=64, lr=3e-4 with linear warmup + cosine decay, weight_decay=0.01, grad_clip=1.0.

## Metrics (real run, seed=6638)
| Metric | Train | Val |
|---|---|---|
| Cross-entropy | 0.8595 | 0.7775 |
| Perplexity | 2.362 | 2.176 |
| Bits/char | 1.240 | 1.122 |

- Generalization gap: -0.0820 (val loss slightly below train loss -- healthy, non-overfit)
- Val top-1 next-char accuracy: 75.45%
- Mean/max gradient norm: 0.966 / 4.99
- NaN/Inf events: 0
- Total training time: 1,726.6s (~28.8 min)
- Avg train tokens/sec: 177,917
- Peak memory: 1,749 MB
- Generation: distinct-1/2/3 = 0.0078 / 0.0535 / 0.1369, repeated 4-gram rate = 0.2056, ~239 tokens/sec

## Hardware
Google Colab GPU runtime (cuda device confirmed in training log).

## Notes
Re-run with personal seed 6638 (replacing an earlier seed=42 run). Full 1.8B-character corpus required vectorizing data.py's character-encoding step (originally a pure-Python loop, impractical at this scale) into a chunked NumPy lookup-table approach -- see src/data.py::encode().
