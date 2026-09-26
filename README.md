# DATA266 Lab 1 — Team 5

**GitHub:** https://github.com/snehas-SJSU/Data_266-Lab1  
**Team:** 5  

| Member | Folder name |
|---|---|
| Sneha Singh | `sneha_singh/` |
| Ritika Mukesh Neema | `ritika_mukesh_neema/` |

Each person works only in their own member folder under every task. Shared data lives in `task*/data/`. Ritika’s folders are placeholders until she adds her work.

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
│   ├── data/                         # shared TinyStories
│   ├── sneha_singh/
│   └── ritika_mukesh_neema/
├── task2_sentiment/
│   ├── data/                         # shared (optional cache)
│   ├── sneha_singh/
│   └── ritika_mukesh_neema/
├── task3_gan/
│   ├── data/monet_jpg/  photo_jpg/   # shared
│   ├── sneha_singh/
│   └── ritika_mukesh_neema/
├── reproducibility/
│   ├── manifests/<member>.md
│   └── raw_logs/<member>/
└── report/
```

---

## Part 1 — LLM (TinyStories)

**Shared data:** `task1_llm/data/` (TinyStories). Drive link in Common table below.

### Sneha Singh — `task1_llm/sneha_singh/`

- `src/part1_llm.ipynb`, `src/config.json`
- `metrics_report.csv`, `results.md`, `failure_analysis.md`
- Smoke checkpoint + loss curve + samples
- Raw log: `reproducibility/raw_logs/sneha_singh/task1_llm/`

**How to run (smoke)**

```bash
# "smoke": true in task1_llm/sneha_singh/src/config.json
jupyter nbconvert --to notebook --execute --inplace \
  task1_llm/sneha_singh/src/part1_llm.ipynb
```

**Full GPU (lab machine — not done yet)**

1. Put TinyStories under `task1_llm/data/` (URL / file name in `config.json`).
2. Set `"smoke": false` in her `src/config.json`.
3. Run notebook on GPU (≥10 epochs, 100K / 10K).
4. Refresh metrics / results / failure analysis and push.

**Status:** smoke done (Mac MPS). Full run still needed on GPU lab.

### Ritika Mukesh Neema — `task1_llm/ritika_mukesh_neema/`

- Folder ready (placeholder). Own architecture / hyperparameters (different from Sneha).
- **Status:** content pending.

---

## Part 2 — Yelp polarity sentiment

**Shared data:** Hugging Face `fancyzhx/yelp_polarity` (each notebook downloads).

### Sneha Singh — `task2_sentiment/sneha_singh/`

- `src/part2_sentiment.ipynb`, `src/config.json`
- 3 models: baseline (mean pool), BiLSTM, TextCNN
- `metrics_report.csv`, `results.md`, `failure_analysis.md` (20 errors)
- Checkpoints: `checkpoints/*_full.pt`
- Confusion matrices + EDA under `outputs/`

**How to run**

```bash
# "smoke": false in her config.json
jupyter nbconvert --to notebook --execute --inplace \
  task2_sentiment/sneha_singh/src/part2_sentiment.ipynb
```

**Status:** full run done (~15k / 3k / 3k, 5 epochs). No re-run needed.

### Ritika Mukesh Neema — `task2_sentiment/ritika_mukesh_neema/`

- Folder ready (placeholder). Own 3 models (different from Sneha).
- **Status:** content pending.

**Team:** model comparison table after Ritika has numbers.

---

## Part 3 — CycleGAN (photo ↔ Monet)

**Shared data:** `task3_gan/data/monet_jpg/`, `photo_jpg/` (class Kaggle). Drive link in Common table below.

### Sneha Singh — `task3_gan/sneha_singh/`

- `src/part3_cyclegan.ipynb`, `src/config.json`, `evaluate_local.py`, `infer_b2a.py`
- `submission.csv` (FID + MiFID for Kaggle)
- `full_metrics_report.csv`, `results.md`, `failure_analysis.md`
- Sample grid + loss curves; 30-image audit under `outputs/human_audit/`
- `checkpoints/` on git is empty of weights; **`best.pt` on Drive** (GitHub 100 MB limit)

**How to run**

- Train: Colab / GPU, `"smoke": false` (done on Tesla T4).
- Local metrics:

```bash
python task3_gan/sneha_singh/evaluate_local.py
```

**Kaggle (Sneha)**

| Item | Value |
|---|---|
| Upload file | `task3_gan/sneha_singh/submission.csv` |
| Local FID / MiFID | 95.13 / 0.411 |
| Public / private / rank | _fill after submit_ |

**Status:** train + full A2B/B2A local eval done (A2B FID 95.13 / MiFID 0.411; B2A FID 96.36 / MiFID 0.433). Open: Drive `best.pt`, Kaggle submit + rank, wait for Ritika’s audit column.

### Ritika Mukesh Neema — `task3_gan/ritika_mukesh_neema/`

- Folder ready (placeholder). Own CycleGAN config / train / `submission.csv`.
- Same 30-image audit sheet: fill `rater_ritika_*` in the shared audit CSV (or her copy under her folder once she runs).
- **Status:** content pending. Kaggle: her own `submission.csv` when ready.

| Item | Value |
|---|---|
| Upload file | `task3_gan/ritika_mukesh_neema/submission.csv` |
| Local FID / MiFID | _add_ |
| Public / private / rank | _fill after submit_ |

**Team:** joint audit agreement (κ) after both rate the 30 samples.

---

## Google Drive

### 1. Common links — 3 shared datasets

Same Drive folder for the team (raw data only):

| # | Dataset | Drive link |
|---|---|---|
| 1 | TinyStories (Part 1) | _add_ |
| 2 | Monet paintings — `monet_jpg` (Part 3) | _add_ |
| 3 | Photos — `photo_jpg` (Part 3) | _add_ |

Optional (not one of the 3 datasets): `real_stats.npz` from Kaggle Data → Drive link _add_ if you download it.

Part 2 Yelp polarity is downloaded from Hugging Face in the notebook — no shared Drive zip required.

### 2. Individual links — Sneha / Ritika

Part 1 and Part 2 checkpoints stay **on git** (they’re small enough).  
Part 3 `best.pt` is ~108 MB → **Drive only** (GitHub 100 MB limit).  
After Part 1 full GPU: check file size; if under 100 MB, commit it under `checkpoints/` like Part 2. Only use Drive if it’s over the limit.

**Sneha Singh**

| File | Drive link |
|---|---|
| Part 3 `best.pt` | _add_ |

**Ritika Mukesh Neema**

| File | Drive link |
|---|---|
| Part 3 `best.pt` | _add_ |

---

## Reproducibility

| Member | Manifest | Raw logs |
|---|---|---|
| Sneha Singh | `reproducibility/manifests/sneha_singh.md` | `reproducibility/raw_logs/sneha_singh/` |
| Ritika Mukesh Neema | `reproducibility/manifests/ritika_mukesh_neema.md` (_add when she runs_) | `reproducibility/raw_logs/ritika_mukesh_neema/` |

---

## Team status

| Part | Sneha Singh | Ritika Mukesh Neema |
|---|---|---|
| 1 LLM | Smoke done; full GPU still needed | Placeholder |
| 2 Yelp | Full run done | Placeholder |
| 3 CycleGAN | Train + full A2B/B2A eval done; Kaggle + Drive + joint audit open | Placeholder |
| Report PDF | Not started | Not started |
