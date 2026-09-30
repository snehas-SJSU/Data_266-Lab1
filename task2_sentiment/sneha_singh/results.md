# Part 2 — Yelp polarity (Sneha Singh)

Dataset: Yelp polarity (not IMDB). Embeddings learned from scratch. No pretrained LMs. The files are not on Drive and not in git. The notebook downloads `fancyzhx/yelp_polarity` from Hugging Face when the run starts.

## Models and why I chose them
1. **Baseline** — mean pooling over learned embeddings + linear. Simple bag-of-embeddings control. If this is already strong, fancier models must beat it clearly.
2. **Experimental A** — BiLSTM over embeddings. Keeps word order, so negation and long reviews can matter more than a mean pool.
3. **Experimental B** — TextCNN over embeddings. Local n-gram filters (3/4/5) catch short phrases like "not good" / "highly recommend".

## Preprocessing
- lowercasing, punctuation removal, English stopwords, WordNet lemmatization
- vocab from training text only; `nn.Embedding` trained with each model
- full Yelp train file: 10,000 reviews held out for validation, up to 550,000 for training, and the official 38,000-row test set
- 47 training reviews were empty after cleaning, so the split used is 549,953 / 10,000 / 38,000
- EDA plot: `outputs/eda.png` (length + class balance)

## Hardware
- baseline: Colab Tesla T4
- experimental_a: Colab Tesla T4
- experimental_b: Colab Tesla T4
- Smoke: False
- Train/val/test sizes: 549953 / 10000 / 38000
- Epochs: 5  |  batch: 64  |  emb_dim: 100  |  max_len: 128

## Test metrics

| Model | Acc | Macro-F1 | ROC-AUC | MCC | Brier | ECE | Time | Params | Peak MB |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (mean pool) | 0.9316 | 0.9316 | 0.9794 | 0.8631 | 0.0516 | 0.0060 | 233 s | 10.68M | 228 |
| BiLSTM | 0.9416 | 0.9416 | 0.9862 | 0.8834 | 0.0438 | 0.0116 | 794 s | 10.76M | 344 |
| TextCNN | 0.9374 | 0.9374 | 0.9840 | 0.8749 | 0.0466 | 0.0075 | 3646 s | 10.75M | 1656 |

The rest of the required metrics (precision/recall/F1 macro, micro, and weighted, PR-AUC, bootstrap intervals, McNemar, short vs long slices) are in `metrics_report.csv`.

## Comparison (my three models)
- Best macro-F1: **BiLSTM** (0.9416). TextCNN is 0.9374. Baseline is 0.9316.
- McNemar vs baseline: BiLSTM p < 0.001, TextCNN p < 0.001. On this 38k test set both experimental models beat the mean pool.
- Short-review macro-F1: BiLSTM 0.947, TextCNN 0.940, baseline 0.934.
- Long-review macro-F1: BiLSTM 0.934, TextCNN 0.933, baseline 0.927.
- Long reviews have a higher error rate for every model (BiLSTM 0.065 long vs 0.052 short).

## Strengths
- Same clean pipeline for all three models, so the bake-off is fair.
- The BiLSTM uses word order and beats the mean-pool baseline on the full test set.
- TextCNN is close behind and much of the gain shows up on short reviews.
- Calibration metrics (Brier, ECE) and McNemar are reported, not only accuracy.

## Weaknesses / limitations
- Train/val/test after cleaning is 549,953 / 10,000 / 38,000.
- max_len=128 truncates long reviews.
- Stopword removal can drop useful words like "not" if cleaning is too aggressive (I already keep short tokens carefully, but negation is still hard).
- The error review is 20 BiLSTM mistakes. Mixed reviews and negation are still the cases it misses.

## Future work
- Keep negation words in the tokenizer (do not drop "not" / "never").
- Try a small attention pooling head on BiLSTM.
- Raise max_len so long reviews are not cut at 128 tokens.
- Add more robustness slices (very short reviews, reviews with "but", foreign-language noise).
