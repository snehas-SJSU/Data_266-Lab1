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

Raw datasets and Part 3 `best.pt` are too large for GitHub (100 MB limit). We keep them on a shared Drive folder and copy them into the paths below on each machine.

**Drive folder layout**

```
DATA266_Lab1_Team5_Drive/
├── shared_data/
│   ├── TinyStories/
│   └── gan/
│       ├── monet_jpg/
│       ├── photo_jpg/
│       └── real_stats.npz
└── sneha_singh/
    └── task3_gan/
        ├── checkpoints/
        │   └── best.pt
        └── images.zip
```

| Item | Copy into local path | Drive link |
|---|---|---|
| Shared Drive (root) | — | _add_ |
| TinyStories | `task1_llm/data/` | _add_ |
| Monet + Photo | `task3_gan/data/monet_jpg/`, `photo_jpg/` | _add_ |
| `real_stats.npz` | `task3_gan/data/real_stats.npz` | _add_ |
| Part 3 `best.pt` | `task3_gan/sneha_singh/checkpoints/best.pt` | _add_ |
| Part 3 `images.zip` | `task3_gan/sneha_singh/images.zip` | _add_ |

Yelp data loads from Hugging Face in the Part 2 notebook.

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
