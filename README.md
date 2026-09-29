# DATA266 Lab 1 — Team 5

**GitHub:** https://github.com/snehas-SJSU/Data_266-Lab1  
**Team:** 5  

| Member | Folder name |
|---|---|
| Sneha Singh | `sneha_singh/` |
| Ritika Mukesh Neema | `ritika_mukesh_neema/` |

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
│   ├── data/                         # not on Drive; notebook downloads Yelp
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

- `src/part1_llm.ipynb`, `src/config.json` (`smoke: false`)
- `metrics_report.csv`, `results.md`, `failure_analysis.md`
- `checkpoints/best.pt` (12 MB, full run)
- Loss curve + samples under `outputs/`
- Raw log: `reproducibility/raw_logs/sneha_singh/task1_llm/train_full.log`

**How to run**

`config.json` is the full run: 100000 / 10000, 10 epochs. The notebook downloads `TinyStories-train.txt` into `task1_llm/data/` if it is missing. That file stays off git.

```bash
jupyter nbconvert --to notebook --execute --inplace \
  task1_llm/sneha_singh/src/part1_llm.ipynb
```

**Status:** full run done (2026-09-28, NVIDIA GeForce RTX 5090, 10 epochs). Val CE 0.857, perplexity 2.36, top-1 next-char accuracy 0.729. Smoke log from the earlier Mac run is still in `train_smoke.log`.

### Ritika Mukesh Neema — `task1_llm/ritika_mukesh_neema/`

Notebook, config, metrics, and write-ups for her Part 1 run.

---

## Part 2 — Yelp polarity sentiment

**Shared data:** not on Drive and not in git. The notebook downloads Hugging Face `fancyzhx/yelp_polarity` itself when it runs, so there is no Yelp zip to upload.

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

**Status:** full run done (100k train sampled, 99,989 after empty-review drop / 10k val / 10k test, 5 epochs). Baseline macro-F1 0.923.

### Ritika Mukesh Neema — `task2_sentiment/ritika_mukesh_neema/`

Notebook, config, metrics, and write-ups for her Part 2 run (her own 3 models).

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

- Train: CUDA GPU, `"smoke": false`, AMP (80 epochs, ~4.3 h, peak ~6568 MB).
- Local metrics:

```bash
python task3_gan/sneha_singh/evaluate_local.py
```

**Kaggle (Sneha)**

| Item | Value |
|---|---|
| Upload file | `task3_gan/sneha_singh/submission.csv` |
| Local FID / MiFID | 89.99 / 0.403 |
| Public / private / rank | _fill after submit_ |

**Status:** Retrain and full A2B/B2A local eval are done (A2B FID 89.99 / MiFID 0.403; B2A FID 92.23 / MiFID 0.423). `best.pt` and the Monet/photo folders are on Drive.

### Ritika Mukesh Neema — `task3_gan/ritika_mukesh_neema/`

Notebook, `evaluate_local.py`, `submission.csv`, metrics, and write-ups for her Part 3 run. Kaggle upload from her own `submission.csv`.

| Item | Value |
|---|---|
| Upload file | `task3_gan/ritika_mukesh_neema/submission.csv` |
| Local FID / MiFID | _add_ |
| Public / private / rank | _fill after submit_ |

---

## Google Drive

### 1. Common links — 3 shared datasets

Same Drive folder for the team (raw data only):

| # | Dataset | Link |
|---|---|---|
| 1 | TinyStories-train.txt (Part 1, 1.92 GB) | https://drive.google.com/file/d/1K8wGXMKDaLaVdv7ISMNihMH33pGig-CN/view?usp=share_link |
| 2 | Monet paintings — `monet_jpg` (Part 3) | https://drive.google.com/drive/folders/1BXYfhW8uZ6umK1TZFW8Un62mZVK72L7Y?usp=share_link |
| 3 | Photos — `photo_jpg` (Part 3) | https://drive.google.com/drive/folders/1BXYfhW8uZ6umK1TZFW8Un62mZVK72L7Y?usp=share_link |

Optional (not one of the 3 datasets): `real_stats.npz` from Kaggle Data → Drive link _add_ if you download it.

Part 2 is not in this table on purpose. Yelp polarity is not a file we host. Each notebook downloads `fancyzhx/yelp_polarity` from Hugging Face at run time.

### 2. Individual links — Sneha / Ritika

Part 1 `best.pt` (12 MB) and Part 2 checkpoints stay **on git**.  
Part 3 `best.pt` is 113 MB, over GitHub’s 100 MB limit, so the checkpoint link is here. The datasets stay in the common table above.

**Sneha Singh**

| File | Link |
|---|---|
| Part 3 `best.pt` | https://drive.google.com/file/d/1TVDCjIGkr5Xt5nrrOmkjhKA3G9ppj-2T/view?usp=share_link |

**Ritika Mukesh Neema**

| File | Link |
|---|---|
| Part 3 `best.pt` | _add_ |

---

## Reproducibility

| Member | Manifest | Raw logs |
|---|---|---|
| Sneha Singh | `reproducibility/manifests/sneha_singh.md` | `reproducibility/raw_logs/sneha_singh/` |
| Ritika Mukesh Neema | `reproducibility/manifests/ritika_mukesh_neema.md` | `reproducibility/raw_logs/ritika_mukesh_neema/` |

---

## Team status

| Part | Sneha Singh | Ritika Mukesh Neema |
|---|---|---|
| 1 LLM | Full GPU run done (val CE 0.857) | — |
| 2 Yelp | Full run done (100k / 10k / 10k). Data is downloaded in the notebook, not on Drive | — |
| 3 CycleGAN | Train + local eval done. `best.pt` on Drive. Kaggle CSV not uploaded yet | — |
| Report PDF | `report/DATA266_Lab1_Report_Team_5.pdf` (Sneha’s sections; Ritika still open) | — |
