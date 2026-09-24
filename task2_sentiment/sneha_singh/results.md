# Part 2 — Yelp polarity (Sneha Singh)

Dataset: Yelp polarity (not IMDB). Embeddings learned from scratch. No pretrained LMs.

## Models and why I chose them
1. **Baseline** — mean pooling over learned embeddings + linear. Simple bag-of-embeddings control. If this is already strong, fancier models must beat it clearly.
2. **Experimental A** — BiLSTM over embeddings. Keeps word order, so negation and long reviews can matter more than a mean pool.
3. **Experimental B** — TextCNN over embeddings. Local n-gram filters (3/4/5) catch short phrases like "not good" / "highly recommend".

## Preprocessing
- lowercasing, punctuation removal, English stopwords, WordNet lemmatization
- vocab from training text only; `nn.Embedding` trained with each model

## Hardware
- Device: mps (Apple Silicon)
- Smoke: False
- Train/val/test sizes: 14998 / 3000 / 3000
- Epochs: 5  |  batch: 64  |  emb_dim: 100  |  max_len: 128

## Test metrics (summary)
- **baseline**: acc=0.9043  macro-F1=0.9043  ROC-AUC=0.9648  MCC=0.8086  Brier=0.0718  ECE=0.0194  time=10.6s  params=1944302
- **experimental_a**: acc=0.8850  macro-F1=0.8849  ROC-AUC=0.9546  MCC=0.7702  Brier=0.0839  ECE=0.0278  time=439.3s  params=2029350
- **experimental_b**: acc=0.8960  macro-F1=0.8960  ROC-AUC=0.9597  MCC=0.7921  Brier=0.0791  ECE=0.0264  time=11.9s  params=2021478

Full table: `metrics_report.csv` (includes P/R/F1 macro/micro/weighted, PR-AUC, bootstrap CIs, McNemar, length slices).

## Comparison (my three models)
- Best macro-F1: **baseline** (0.9043). Second: **experimental_b** (0.8960).
- McNemar p (experimental_a vs baseline): 0.0009602499332239026
- McNemar p (experimental_b vs baseline): 0.1298061984600565
- Length slices: short-review macro-F1 best=baseline; long-review macro-F1 best looks at CSV columns macro_f1_short / macro_f1_long.

## Strengths
- Same clean pipeline for all three models, so the bake-off is fair.
- Baseline is fast and still competitive, which is useful as a sanity check.
- TextCNN is strong on short opinion phrases; BiLSTM is there to test order-sensitive cases.
- Calibration metrics (Brier, ECE) and McNemar are reported, not only accuracy.

## Weaknesses / limitations
- Smoke/full size is still a subset of full Yelp polarity; more data should help rare phrases.
- max_len=128 truncates long reviews.
- Stopword removal can drop useful words like "not" if cleaning is too aggressive (I already keep short tokens carefully, but negation is still hard).
- Peak memory on MPS/CPU is reported as 0 in CSV; CUDA would fill that field.
- Error review is automatic sampling of hard cases; a human pass on wording can still refine error types.

## Future work
- Keep negation words in the tokenizer (do not drop "not" / "never").
- Try a small attention pooling head on BiLSTM.
- Tune CNN filter counts / kernel set, and train longer on the full Yelp train split.
- Add more robustness slices (very short reviews, reviews with "but", foreign-language noise).
