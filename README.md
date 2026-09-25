# DATA266 Lab 1 — Team 5

**GitHub:** https://github.com/snehas-SJSU/Data_266-Lab1  
**Team:** 5  
**Members:** Sneha Singh (`sneha_singh/`), Ritika Mukesh Neema (`ritika_mukesh_neema/`)

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

## Google Drive

GitHub has the code and small files. Large files each get their **own Drive link** (share → Anyone with the link → Viewer). Paste that link in the last column.

| # | File / folder | Local path after download | Individual Drive link |
|---|---|---|---|
| 1 | TinyStories data | `task1_llm/data/` | _add_ |
| 2 | Monet images (`monet_jpg`) | `task3_gan/data/monet_jpg/` | _add_ |
| 3 | Photo images (`photo_jpg`) | `task3_gan/data/photo_jpg/` | _add_ |
| 4 | `real_stats.npz` | `task3_gan/data/real_stats.npz` | _add_ |
| 5 | **`best.pt`** (Part 3 full weights) | **`task3_gan/sneha_singh/checkpoints/best.pt`** | **_add_** |
| 6 | `images.zip` (Part 3 preds) | `task3_gan/sneha_singh/images.zip` | _add_ |

Yelp loads from Hugging Face in the Part 2 notebook (no Drive file required).

**Drive folder names (for upload organization)**

```
DATA266_Lab1_Team5_Drive/
├── 01_tinystories/
├── 02_monet_jpg/
├── 03_photo_jpg/
├── 04_real_stats.npz
├── 05_sneha_best.pt          ← upload best.pt here, then paste link in row 5
└── 06_sneha_images.zip
```

---

## Repo layout

Same structure as the lab PDF:

```
Data_266-Lab1/
├── README.md
├── requirements.txt
├── task1_llm/data/ + sneha_singh/ + ritika_mukesh_neema/
├── task2_sentiment/data/ + sneha_singh/ + ritika_mukesh_neema/
├── task3_gan/data/{monet_jpg,photo_jpg}/ + sneha_singh/ + ritika_mukesh_neema/
├── reproducibility/{manifests,raw_logs}/
└── report/
```

Each member folder:

```
src/  data_processed/  checkpoints/  outputs/
metrics_report.csv   (Parts 1–2)
failure_analysis.md
results.md
```

Part 3 also has `evaluate_local.py`, `submission.csv`, and `full_metrics_report.csv`.

---

## Results

| Part | Sneha path |
|---|---|
| 1 LLM | `task1_llm/sneha_singh/` |
| 2 Sentiment | `task2_sentiment/sneha_singh/` |
| 3 CycleGAN | `task3_gan/sneha_singh/` |
| Manifest | `reproducibility/manifests/sneha_singh.md` |
| Logs | `reproducibility/raw_logs/sneha_singh/` |

---

## Smoke test

Set `"smoke": true` in `src/config.json`, then:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  task1_llm/sneha_singh/src/part1_llm.ipynb
```

```bash
jupyter nbconvert --to notebook --execute --inplace \
  task2_sentiment/sneha_singh/src/part2_sentiment.ipynb
```

Part 3 full train: Colab, Tesla T4, 80 epochs. Metrics:

```bash
python task3_gan/sneha_singh/evaluate_local.py
```

Kaggle upload file: `task3_gan/sneha_singh/submission.csv` (FID + MiFID).

---

## Status

| Part | Sneha | Ritika |
|---|---|---|
| 1 LLM | Smoke done | Folder ready |
| 2 Yelp | Full run done | Folder ready |
| 3 CycleGAN | Train + local eval done; Kaggle submit + Drive links still open | Folder ready |
| Report PDF | Not started | Not started |
