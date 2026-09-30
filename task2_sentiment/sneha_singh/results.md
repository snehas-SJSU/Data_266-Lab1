# Part 2 — Yelp polarity (Sneha Singh)

Dataset: Yelp polarity (not IMDB). Embeddings learned from scratch. No pretrained LMs. The files are not on Drive and not in git. The notebook downloads `fancyzhx/yelp_polarity` from Hugging Face when the run starts.

## Models and why I chose them
1. **Baseline** — mean pooling over learned embeddings + linear. Simple bag-of-embeddings control. If this is already strong, fancier models must beat it clearly.
2. **Experimental A** — BiLSTM over embeddings. Keeps word order, so negation and long reviews can matter more than a mean pool.
3. **Experimental B** — TextCNN over embeddings. Local n-gram filters (3/4/5) catch short phrases like "not good" / "highly recommend".

## Preprocessing
- lowercasing, punctuation removal, English stopwords, WordNet lemmatization
- vocab from training text only; `nn.Embedding` trained with each model
- 100,000 train reviews sampled; 11 empty after cleaning, so 99,989 / 10,000 / 10,000 were used
- EDA plot: `outputs/eda.png` (length + class balance)

## Hardware
- baseline: Apple M4 (10 cores: 4 performance, 6 efficiency), 16 GB unified, MPS
- experimental_a: Apple M4 (10 cores: 4 performance, 6 efficiency), 16 GB unified, MPS
- experimental_b: Apple M4 (10 cores: 4 performance, 6 efficiency), 16 GB unified, MPS
- Smoke: False
- Train/val/test sizes: 99989 / 10000 / 10000
- Epochs: 5  |  batch: 64  |  emb_dim: 100  |  max_len: 128

## Test metrics

| Model | Acc | Macro-F1 | ROC-AUC | MCC | Brier | ECE | Time | Params |
|---|---|---|---|---|---|---|---|---|
| Baseline (mean pool) | 0.9226 | 0.9226 | 0.9729 | 0.8452 | 0.0595 | 0.0115 | 114 s | 4.76M |
| BiLSTM | 0.9184 | 0.9184 | 0.9752 | 0.8370 | 0.0602 | 0.0169 | 1706 s | 4.85M |
| TextCNN | 0.9207 | 0.9207 | 0.9747 | 0.8414 | 0.0607 | 0.0241 | 108 s | 4.84M |

The rest of the required metrics (precision/recall/F1 macro, micro, and weighted, PR-AUC, bootstrap intervals, McNemar, short vs long slices) are in `metrics_report.csv`.

## Comparison (my three models)
- Best macro-F1: **baseline** (0.9226). TextCNN is 0.9207. BiLSTM is 0.9184.
- McNemar vs baseline: BiLSTM p=0.091, TextCNN p=0.424. On this 10k test set, neither experimental model is clearly different from the baseline.
- Short-review macro-F1: baseline 0.925, TextCNN 0.924, BiLSTM 0.922.
- Long-review macro-F1: baseline 0.917, TextCNN 0.915, BiLSTM 0.912.
- Long reviews have a higher error rate for every model (baseline 0.081 long vs 0.074 short).

## Strengths
- Same clean pipeline for all three models, so the bake-off is fair.
- Baseline is fast and still competitive, which is useful as a sanity check.
- TextCNN is strong on short opinion phrases; BiLSTM is there to test order-sensitive cases.
- Calibration metrics (Brier, ECE) and McNemar are reported, not only accuracy.

## Weaknesses / limitations
- Train size is 100k of the 560k Yelp train split, with 10k val and 10k test. Not the entire corpus.
- max_len=128 truncates long reviews.
- Stopword removal can drop useful words like "not" if cleaning is too aggressive (I already keep short tokens carefully, but negation is still hard).
- Peak memory is 0.0 because this run is MPS. torch.cuda.max_memory_allocated() stays 0, and I am not guessing a number.
- Error review is automatic sampling of hard cases; a human pass on wording can still refine error types.

## Future work
- Keep negation words in the tokenizer (do not drop "not" / "never").
- Try a small attention pooling head on BiLSTM.
- Tune CNN filter counts / kernel set, and train longer on the full Yelp train split.
- Add more robustness slices (very short reviews, reviews with "but", foreign-language noise).
