# Task 2 — Results: Yelp Polarity Sentiment Classification Bake-off

**Author:** Ritika Mukesh Neema

## Models & justification
1. **Baseline — Mean-pooled embeddings + linear head** (`src/models.py::BaselineMeanEmbed`)
   Embeddings learned from scratch, averaged over the sequence (order-blind),
   fed to a single linear layer. Justification: the simplest possible
   from-scratch-embedding baseline — establishes the floor a smarter
   architecture needs to beat, and isolates how much of the signal in Yelp
   Polarity is just "which words appear" vs. "in what order."
2. **Experimental #1 — BiLSTM** (`src/models.py::BiLSTMClassifier`)
   Adds recurrence and directionality: negation and intensifiers ("not
   good", "not bad at all") change meaning based on word order, which the
   mean-pooled baseline structurally cannot see.
3. **Experimental #2 — TextCNN** (`src/models.py::TextCNNClassifier`, Kim 2014
   style, kernel sizes 3/4/5, global max-pool)
   A different inductive bias from both: local n-gram phrase detectors
   instead of a left-to-right recurrent state. Cheaper to train than the
   BiLSTM and often competitive on short-to-medium review text.

All three use embeddings trained end-to-end with the classifier — no
pretrained word2vec/GloVe/transformer embeddings anywhere in the pipeline.

## Preprocessing
Lowercasing, HTML/URL stripping, punctuation removal, stopword removal
(`sklearn.feature_extraction.text.ENGLISH_STOP_WORDS`, no network download
needed), and Porter stemming (`nltk.stem.PorterStemmer` — chosen over
WordNet lemmatization specifically because it's a pure algorithm with no
downloaded corpus dependency, which keeps the pipeline reproducible with
no extra network access at run time).

## How to reproduce
```
cd src
python run_all.py --train_csv <path to yelp train csv> --test_csv <path to yelp test csv>
```
Smoke test: add `--smoke_test`.

## Metrics (full run, no smoke-test subsampling)

| Model    | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC    | Brier  | ECE    | Params    | Train time (s) | Examples/sec | Peak mem (MB) |
|----------|----------|----------|---------|--------|--------|--------|--------|-----------|-----------------|--------------|----------------|
| Baseline | 0.9217   | 0.9217   | 0.9741  | 0.9726 | 0.8434 | 0.0586 | 0.0067 | 3,000,101 | 124.9           | 20,181       | 2,123          |
| BiLSTM   | 0.9337   | 0.9337   | 0.9818  | 0.9821 | 0.8675 | 0.0513 | 0.0253 | 3,235,777 | 797.5           | 3,159        | 2,274          |
| TextCNN  | 0.9260   | 0.9260   | 0.9791  | 0.9794 | 0.8523 | 0.0553 | 0.0214 | 3,120,601 | 267.2           | 9,432        | 2,450          |

Full metric suite (macro/micro/weighted F1, precision/recall macro, 95%
bootstrap CIs, per-length-slice breakdown) is in
`outputs/metrics_report.csv` and `outputs/full_metrics_report.json`.

## Paired McNemar test (baseline vs. each experimental model)
From `outputs/mcnemar_results.json` (b = baseline-right/model-wrong,
c = baseline-wrong/model-right):

- **Baseline vs. BiLSTM:** b=817, c=1275, statistic=99.832, p<0.001 --
  highly significant. BiLSTM fixes far more baseline errors than it
  introduces.
- **Baseline vs. TextCNN:** b=903, c=1068, statistic=13.646, p=0.00022 --
  significant at the 0.05 level, though a much smaller effect than
  BiLSTM's.

Both experimental models are statistically significant improvements over
the mean-pooled baseline, confirming that word order and local structure
carry real signal beyond bag-of-words.

## Manual error review (20 cases)
See `failure_analysis.md` (auto-generated skeleton from
`src/error_analysis.py`, hand-annotated with proposed fixes).

## Comparative analysis
BiLSTM wins on every headline metric (accuracy, macro-F1, ROC-AUC, PR-AUC,
MCC) but costs ~6.4x the baseline's training time and is roughly 3x slower
per-example than TextCNN (797.5s vs 267.2s vs 124.9s), a direct consequence
of its inherently sequential recurrence versus TextCNN's parallelizable
convolutions and the baseline's simple mean pooling. TextCNN is the best
throughput/accuracy tradeoff of the three: within 0.8 points of accuracy
of BiLSTM at roughly a third of the training time and ~3x BiLSTM's
examples/sec. Both McNemar tests confirm each experimental model is a
statistically significant improvement over the baseline, so word order
carries real signal beyond bag-of-words -- but the marginal accuracy gain
from BiLSTM's full recurrence over TextCNN's local n-gram detectors
(93.37% vs 92.60%) is modest relative to the near-6x extra compute cost.
The manual error review shows all three models share the same weak spots:
sarcasm/rhetorical negation, negative-surface-words-inside-a-positive-review,
and long reviews where the sentiment-bearing content sits past the point
`max_len=200` truncation may be cutting.

## Hardware disclosure
Google Colab (free tier), CUDA GPU (Tesla T4), full run on the complete
560,000-row Yelp Polarity dataset (no smoke-test subsampling). Peak memory
per model: baseline 2,123 MB, bilstm 2,274 MB, textcnn 2,450 MB (see
`outputs/metrics_report.csv`).
