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

## Smoke test (one command)

```bash
bash run_smoke.sh
```

Runs Sneha's Part 1 GPT with `smoke: true` (TinyStories-valid, 256 train / 64 val, 2 epochs, about 1 minute on CPU). The notebook downloads the small TinyStories-valid file itself. The script works in a temporary copy of the repo, so committed results, raw logs, and manifests are not touched. It prints the output folder when it finishes.

---

## How to run — all members

Run every command from the repo root after Setup. Each `config.json` is set to the full run (`"smoke": false`). Set `"smoke": true` for a short run.

| Part | Member | Command | Data |
|---|---|---|---|
| 1 — LLM | Sneha | `jupyter nbconvert --to notebook --execute --inplace task1_llm/sneha_singh/src/part1_llm.ipynb` | Notebook downloads TinyStories |
| 1 — LLM | Ritika | `cd task1_llm/ritika_mukesh_neema/src && python run_all.py --train_txt <train.txt> --val_txt <val.txt>` | Her 100K / 10K TinyStories split |
| 2 — Sentiment | Sneha | `jupyter nbconvert --to notebook --execute --inplace task2_sentiment/sneha_singh/src/part2_sentiment.ipynb` | Notebook downloads Yelp polarity |
| 2 — Sentiment | Ritika | `cd task2_sentiment/ritika_mukesh_neema/src && python run_all.py --train_csv <train.csv> --test_csv <test.csv>` | Her Yelp train / test CSVs |
| 3 — CycleGAN | Sneha | `jupyter nbconvert --to notebook --execute --inplace task3_gan/sneha_singh/src/part3_cyclegan.ipynb` | `monet_jpg/`, `photo_jpg/` from Drive into `task3_gan/data/` |
| 3 — CycleGAN | Ritika | `cd task3_gan/ritika_mukesh_neema/src && python run_all.py` | Same as above |

Part 3 scoring (professor evaluation script, both directions averaged, writes `submission.csv`):

```bash
python task3_gan/sneha_singh/evaluate_local.py
```

Part 3 trains on a CUDA GPU. Parts 1 and 2 also run on CPU or Apple MPS, only slower. Ritika's `run_all.py` scripts also accept `--smoke_test`.

---

## Repo layout

```
Data_266-Lab1/
├── README.md
├── run_smoke.sh                      # one-command smoke test
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

**Status:** full run done (549,953 train / 10k val / 38k test, 5 epochs, Colab Tesla T4). Best model is the BiLSTM, macro-F1 0.942.

### Ritika Mukesh Neema — `task2_sentiment/ritika_mukesh_neema/`

Notebook, config, metrics, and write-ups for her Part 2 run (her own 3 models).

---

## Part 3 — CycleGAN (photo ↔ Monet)

**Shared data:** `task3_gan/data/monet_jpg/`, `photo_jpg/` (class Kaggle). Drive link in Common table below.

### Sneha Singh — `task3_gan/sneha_singh/`

- `src/part3_cyclegan.ipynb`, `src/config.json`, `evaluate_local.py`, `infer_b2a.py`
- `submission.csv` (FID + MiFID for Kaggle)
- `full_metrics_report.csv`, `metrics_report.csv` (long format: final metrics, human audit, per-epoch losses, scored checkpoints), `results.md`, `failure_analysis.md`
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
| Local FID / MiFID | 107.20 / 0.415 (average of both directions) |

**Status:** Professor eval is in `submission.csv` (FID 107.20, MiFID 0.415). A2B Monet→photo FID 109.33. B2A photo→Monet FID 105.06. `best.pt` is the 80-epoch checkpoint. Later retrains scored worse and are not the upload.

### Ritika Mukesh Neema — `task3_gan/ritika_mukesh_neema/`

Notebook, `evaluate_local.py`, `submission.csv`, metrics, and write-ups for her Part 3 run. Kaggle upload from her own `submission.csv`.

| Item | Value |
|---|---|
| Upload file | `task3_gan/ritika_mukesh_neema/submission.csv` |
| Local FID (training-split, not official) | 123.70 |
| Kaggle official FID / MiFID | 104.30 / 0.413 |
| Public score | -52.3574 (own submission, own Kaggle account) |

---

## Google Drive

### 1. Common links — 3 shared datasets

Same Drive folder for the team (raw data only):

| # | Dataset | Link |
|---|---|---|
| 1 | TinyStories-train.txt (Part 1). On Drive because the file is 1.79 GB, over GitHub’s 100 MB limit. | https://drive.google.com/drive/folders/12MFVOo6T3QRW3X6THiDkh5svtNgL13W_?usp=share_link |
| 2 | Monet paintings — `monet_jpg` (Part 3) | https://drive.google.com/drive/folders/1BXYfhW8uZ6umK1TZFW8Un62mZVK72L7Y?usp=share_link |
| 3 | Photos — `photo_jpg` (Part 3) | https://drive.google.com/drive/folders/1BXYfhW8uZ6umK1TZFW8Un62mZVK72L7Y?usp=share_link |

Part 2 is not in this table. Yelp polarity is not a file we host. Each notebook downloads `fancyzhx/yelp_polarity` from Hugging Face at run time.

### 2. Individual links — Sneha / Ritika

Part 1 `best.pt` (12 MB) and Part 2 checkpoints stay **on git**.  
Part 3 `best.pt` is on Drive because the file is 107.9 MB, over GitHub’s 100 MB limit. The datasets stay in the common table above.

**Sneha Singh**

| File | Link |
|---|---|
| Part 3 `best.pt` (107.9 MB, over the 100 MB git limit) | https://drive.google.com/drive/folders/12AgM95RbZAQUyouH7sukM9nu_rVTD3C9?usp=share_link |

**Ritika Mukesh Neema**

| File | Link |
|---|---|
| Part 3 `best.pt` | https://drive.google.com/file/d/1Wlu22EKbp4QATijRbNBbDOFFGHLHUiNj/view |

---

## Reproducibility

| Member | Manifest | Raw logs |
|---|---|---|
| Sneha Singh | `reproducibility/manifests/sneha_singh.md` | `reproducibility/raw_logs/sneha_singh/` |
| Ritika Mukesh Neema | `reproducibility/manifests/ritika_mukesh_neema.md` | `reproducibility/raw_logs/ritika_mukesh_neema/` |
