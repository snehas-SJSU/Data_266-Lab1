# Part 1 — GPT from scratch (Sneha Singh)

## Architecture
- Tokenizer: character-level (`char_to_idx` / `idx_to_char`)
- Smoke split: 256 train / 64 val (seed=670)
- Full split (when I run it): 100K train / 10K val
- n_layer=4, n_head=4, n_embd=256, block_size=128
- Causal mask: yes
- No `nn.Transformer` / no prebuilt attention

## Hyperparameters
See `src/config.json`.
- Smoke: 2 epochs, warmup 4, batch 32, lr=0.001, AdamW
- Full run target: ≥10 epochs on 100K / 10K

## Metrics
Current numbers in `metrics_report.csv` are from the **smoke** run on Mac (MPS).
- Last train CE: 3.0619
- Last val CE: 3.0660
- Perplexity: 21.4551
- Top-1 next-char acc: 0.1949

All PDF metrics columns are in that CSV (CE, perplexity, BPC, gap, accuracy, distinct-n, repeated 4-gram, grad norm, params, tokens/sec, time).

## Hardware
- Smoke: Mac MPS (Apple Silicon)
- Full GPU run: not done yet (planned on campus GPU lab)

## GPU lab checklist (when I run full)
1. Clone / pull this repo on the lab machine.
2. Download TinyStories into `task1_llm/data/` (file name in `config.json`: `TinyStories-valid.txt`; ~2 GB if using the large split the lab expects).
3. In `src/config.json` set `"smoke": false` (keeps `n_train_full=100000`, `n_val_full=10000`, `min_epochs=10`).
4. Run `src/part1_llm.ipynb` with CUDA.
5. Replace smoke metrics: `metrics_report.csv`, loss curve, samples, 3 failure cases in `failure_analysis.md`, then update this file’s Metrics / Hardware sections and push.

## Notes
- 4 layers / 4 heads / 256 dim is small enough for TinyStories but still a real multi-block GPT.
- Character-level matches the lab (no BPE).
- Causal triu mask blocks future tokens in training and generation.
- Config stays `"smoke": true` on git until the full GPU run finishes — flip only on the lab machine.
