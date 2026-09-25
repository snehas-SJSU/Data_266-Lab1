# Part 2 — Yelp polarity (Sneha Singh)

Dataset: Yelp polarity (not IMDB). Embeddings learned from scratch. No pretrained LMs.

## Models
1. **Baseline** — mean pooling over learned embeddings + linear. Simple control so the other two have to beat something real.
2. **Experimental A** — BiLSTM over embeddings. Keeps word order (negation, long reviews).
3. **Experimental B** — TextCNN over embeddings. Local n-gram filters (3/4/5) for short phrases.

## Preprocessing
- lowercasing, punctuation removal, English stopwords, WordNet lemmatization
- vocab from training text only; `nn.Embedding` trained with each model
- EDA plot: `outputs/eda.png` (length + class balance)

## Hardware
- Device: mps (Apple Silicon)
- Smoke: False (full run)
- Train/val/test: ~15000 / 3000 / 3000
- Epochs: 5  |  batch: 64  |  emb_dim: 100  |  max_len: 128

## Test metrics (summary)
- **baseline**: acc=0.9043  macro-F1=0.9043  ROC-AUC=0.9648  MCC=0.8086  Brier=0.0718  ECE=0.0194  time=10.6s
- **experimental_a**: acc=0.8850  macro-F1=0.8849  ROC-AUC=0.9546  MCC=0.7702  Brier=0.0839  ECE=0.0278  time=439.3s
- **experimental_b**: acc=0.8960  macro-F1=0.8960  ROC-AUC=0.9597  MCC=0.7921  Brier=0.0791  ECE=0.0264  time=11.9s

Full table: `metrics_report.csv` (P/R/F1 macro/micro/weighted, PR-AUC, bootstrap CIs, McNemar, length slices, params, CMs).

## Comparison (my three)
- Best macro-F1: **baseline** (0.9043). Second: TextCNN (0.8960). BiLSTM was slowest and a bit behind.
- McNemar vs baseline: A p≈0.001 (different), B p≈0.13 (not clearly different).
- Short vs long slices: see `macro_f1_short` / `macro_f1_long` in the CSV.

## Strengths
- Same pipeline for all three, so the bake-off is fair.
- Baseline is fast and strong.
- TextCNN is good on short opinion phrases; BiLSTM tests order.
- Calibration (Brier, ECE) and McNemar are reported, not only accuracy.

## Weaknesses
- Full run still uses a subset of Yelp (15k train), not the entire corpus.
- max_len=128 cuts long reviews.
- Stopwords can hurt negation (`not` / `never`).
- Peak memory shows 0 on MPS; would be filled on CUDA.
- Teammate comparison waits on Ritika’s numbers.

## Future work
- Keep negation words in the tokenizer.
- Small attention pool on BiLSTM.
- More data / longer train.
- Extra slices (reviews with "but", very short text).
