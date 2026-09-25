# DATA266 Lab 1 — Team 5

**GitHub:** https://github.com/snehas-SJSU/Data_266-Lab1  
**Team number:** 5  
**Members:**
- Sneha Singh → `sneha_singh/`
- Ritika Mukesh Neema → `ritika_mukesh_neema/`

This README covers **setup**, **how to reproduce a smoke test**, and **where results live** (Lab Section 1 & 5).

No personal file paths, credentials, or API keys are committed. Paths in notebooks/logs are repo-relative (`task1_llm/...`). Runs are config-driven via each member’s `src/config.json`.

---

## 1. Setup

```bash
git clone https://github.com/snehas-SJSU/Data_266-Lab1.git
cd Data_266-Lab1
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional: create a Jupyter kernel named `hw1-data266` if you use that name in Part 2 docs.

**Datasets (not in GitHub):** raw data stays on Google Drive / Hugging Face. Keep only empty `data/` placeholders in git.

| Dataset | Local folder | How to get it | Drive link |
|---|---|---|---|
| TinyStories | `task1_llm/data/` | notebook downloads, or place `TinyStories-valid.txt` here | _add_ |
| Yelp polarity | `task2_sentiment/data/` | auto via Hugging Face `fancyzhx/yelp_polarity` | _add_ |
| Monet / Photo | `task3_gan/data/monet_jpg/`, `photo_jpg/` | Kaggle competition data | _add_ |

Shared Drive folder: _add_ (Anyone with the link → Viewer)

---

## 2. Repository layout

```
Data_266-Lab1/
├── README.md
├── requirements.txt
├── task1_llm/
│   ├── data/                      ← shared TinyStories (not pushed)
│   ├── sneha_singh/
│   └── ritika_mukesh_neema/
├── task2_sentiment/
│   ├── data/                      ← shared Yelp (not pushed; HF cache OK)
│   ├── sneha_singh/
│   └── ritika_mukesh_neema/
├── task3_gan/
│   ├── data/
│   │   ├── monet_jpg/
│   │   └── photo_jpg/
│   ├── sneha_singh/
│   └── ritika_mukesh_neema/
├── reproducibility/
│   ├── manifests/
│   └── raw_logs/
└── report/
    └── DATA266_Lab1_Report_Team_5.pdf
```

### Per member (under each task)

```
<member_name>/
  src/                 executed task notebooks + config.json
  data_processed/      that member’s preprocessing only (not shared)
  checkpoints/         trained weights (.pt / .pth)
  outputs/             plots, samples, predictions, confusion matrices
  metrics_report.csv   all required metrics for this task
  failure_analysis.md  required failure / error write-up
  results.md           architecture + hyperparameter justification
```

Part 3 also uses `evaluate_local.py`, `submission.csv`, and `full_metrics_report.csv` (added when CycleGAN work starts).

---

## 3. Where results live

| What | Path |
|---|---|
| Part 1 notebook / metrics / results | `task1_llm/<member>/` |
| Part 2 notebook / metrics / results | `task2_sentiment/<member>/` |
| Part 3 CycleGAN + Kaggle files | `task3_gan/<member>/` |
| Raw training logs | `reproducibility/raw_logs/<member>/` |
| Run manifests (env + checkpoint map) | `reproducibility/manifests/<member>.md` |
| Combined team report | `report/DATA266_Lab1_Report_Team_5.pdf` |

Example (Sneha, Part 2 full run):
- Metrics: `task2_sentiment/sneha_singh/metrics_report.csv`
- Checkpoints: `task2_sentiment/sneha_singh/checkpoints/*_full.pt`
- Manifest: `reproducibility/manifests/sneha_singh.md`

---

## 4. Reproduce a smoke test (grader command)

Set `"smoke": true` in that member’s `src/config.json`, then run **one** of:

**Part 1 — TinyStories GPT (recommended smoke):**

```bash
jupyter nbconvert --to notebook --execute --inplace \
  task1_llm/sneha_singh/src/part1_llm.ipynb
```

**Part 2 — Yelp sentiment:**

```bash
jupyter nbconvert --to notebook --execute --inplace \
  task2_sentiment/sneha_singh/src/part2_sentiment.ipynb
```

Ritika’s runs use the same commands with `ritika_mukesh_neema` instead of `sneha_singh` once her notebooks are present.

Full runs: set `"smoke": false` in `config.json` (Part 1 prefers GPU/Colab; Part 2 is fine on CPU/MPS).

---

## 5. Reproducibility notes

- Raw logs under `reproducibility/raw_logs/` are left unedited after each run.
- Each member appends a section to `reproducibility/manifests/<member>.md` (versions, device, checkpoints, reproduce command).
- GPU lab machines are wiped after sessions — copy checkpoints and logs to this repo before leaving.

---

## 6. Status (Team 5)

| Part | Sneha | Ritika |
|---|---|---|
| 1 LLM | Code + smoke; full GPU/Colab pending | Folder ready |
| 2 Yelp | Full run complete | Folder ready |
| 3 CycleGAN | Full Colab train + local eval; Kaggle submit pending | Folder ready |
| Report PDF | Pending | Pending |
