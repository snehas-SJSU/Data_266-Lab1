# DATA266 Lab 1 — Team 5

**GitHub:** https://github.com/snehas-SJSU/Data_266-Lab1  
**Team:** 5  
**Members:**
- Sneha Singh → `sneha_singh/`
- Ritika Mukesh Neema → `ritika_mukesh_neema/`

Each person works in their own member folder under every task. Ritika’s folders are placeholders until she adds her work.

---

## Setup

```bash
git clone https://github.com/snehas-SJSU/Data_266-Lab1.git
cd Data_266-Lab1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Repo layout

```
Data_266-Lab1/
├── README.md
├── requirements.txt
├── task1_llm/
│   ├── data/
│   ├── sneha_singh/
│   └── ritika_mukesh_neema/
├── task2_sentiment/
│   ├── data/
│   ├── sneha_singh/
│   └── ritika_mukesh_neema/
├── task3_gan/
│   ├── data/monet_jpg/  photo_jpg/
│   ├── sneha_singh/
│   └── ritika_mukesh_neema/
├── reproducibility/
│   ├── manifests/
│   └── raw_logs/
└── report/
```

---

## Part 1 — LLM (TinyStories)

**Folders**
- Sneha: `task1_llm/sneha_singh/`
- Ritika: `task1_llm/ritika_mukesh_neema/` (placeholder)

**What’s in Sneha’s folder**
- `src/part1_llm.ipynb`, `src/config.json`
- `metrics_report.csv`, `results.md`, `failure_analysis.md`
- `checkpoints/best_smoke.pt`, smoke loss curve + samples
- Raw log: `reproducibility/raw_logs/sneha_singh/task1_llm/`

**How to run (smoke)**

```bash
# set "smoke": true in task1_llm/sneha_singh/src/config.json
jupyter nbconvert --to notebook --execute --inplace \
  task1_llm/sneha_singh/src/part1_llm.ipynb
```

**Status**
- Sneha: smoke done (Mac MPS). Full run still needed (100K / 10K, ≥10 epochs).
- Ritika: folder ready, content pending.

**Data:** `task1_llm/data/` (TinyStories). Drive link below if not downloaded via notebook.

---

## Part 2 — Yelp polarity sentiment

**Folders**
- Sneha: `task2_sentiment/sneha_singh/`
- Ritika: `task2_sentiment/ritika_mukesh_neema/` (placeholder)

**What’s in Sneha’s folder**
- `src/part2_sentiment.ipynb`, `src/config.json`
- 3 models: baseline (mean pool), BiLSTM, TextCNN
- `metrics_report.csv`, `results.md`, `failure_analysis.md` (20 errors)
- Checkpoints: `checkpoints/*_full.pt`
- Confusion matrices + EDA under `outputs/`

**How to run**

```bash
# full run uses "smoke": false in config.json
jupyter nbconvert --to notebook --execute --inplace \
  task2_sentiment/sneha_singh/src/part2_sentiment.ipynb
```

**Status**
- Sneha: full run done (~15k / 3k / 3k, 5 epochs). No re-run needed.
- Ritika: folder ready, content pending.
- Team model comparison: after Ritika has numbers.

**Data:** Hugging Face `fancyzhx/yelp_polarity` (notebook download).

---

## Part 3 — CycleGAN (photo ↔ Monet)

**Folders**
- Sneha: `task3_gan/sneha_singh/`
- Ritika: `task3_gan/ritika_mukesh_neema/` (placeholder)

**What’s in Sneha’s folder**
- `src/part3_cyclegan.ipynb`, `src/config.json`
- `evaluate_local.py`
- `submission.csv` (FID + MiFID for Kaggle)
- `full_metrics_report.csv`, `results.md`, `failure_analysis.md`
- Sample grid + loss curves; 30-image audit under `outputs/human_audit/`
- `checkpoints/` folder on git; **`best.pt` on Drive** (GitHub 100 MB limit)

**How to run**
- Train: Colab / GPU, `"smoke": false` in config (already done on Tesla T4).
- Local metrics:

```bash
python task3_gan/sneha_singh/evaluate_local.py
```

**Kaggle**
- Upload: `task3_gan/sneha_singh/submission.csv`
- Local scores: FID 95.13, MiFID 0.411
- Public / private / rank: fill after submit

**Status**
- Sneha: train + local eval done. Open: Drive `best.pt` link, Kaggle submit + rank, Ritika audit column, full B2A metrics (optional).
- Ritika: folder ready, content pending.

**Data:** `task3_gan/data/monet_jpg/`, `photo_jpg/` from class Kaggle dataset.

---

## Google Drive (individual links)

Each large file gets its own share link. Paste into the last column.

| # | File / folder | Local path | Drive link |
|---|---|---|---|
| 1 | TinyStories | `task1_llm/data/` | _add_ |
| 2 | Monet (`monet_jpg`) | `task3_gan/data/monet_jpg/` | _add_ |
| 3 | Photos (`photo_jpg`) | `task3_gan/data/photo_jpg/` | _add_ |
| 4 | `real_stats.npz` | `task3_gan/data/real_stats.npz` | _add_ |
| 5 | Part 3 **`best.pt`** | `task3_gan/sneha_singh/checkpoints/best.pt` | _add_ |
| 6 | Part 3 `images.zip` | `task3_gan/sneha_singh/images.zip` | _add_ |

---

## Reproducibility

- Manifests: `reproducibility/manifests/sneha_singh.md` (Ritika: `ritika_mukesh_neema.md` when she adds runs)
- Raw logs: `reproducibility/raw_logs/<member>/`

---

## Team status

| Part | Sneha | Ritika |
|---|---|---|
| 1 LLM | Smoke done; full GPU still needed | Placeholder |
| 2 Yelp | Full run done | Placeholder |
| 3 CycleGAN | Train + eval done; Kaggle + Drive + joint audit open | Placeholder |
| Report PDF | Not started | Not started |
