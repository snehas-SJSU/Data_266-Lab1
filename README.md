# DATA266 Lab 1 — Sneha Singh

**GitHub repo:** https://github.com/snehas-SJSU/Data_266-Lab1  
**Team number:** _add from Canvas_  
**Members:** Sneha Singh (`sneha_singh/`), Ritika Mukesh Neema (`ritika_mukesh_neema/`)

No personal file paths, credentials, or API keys in committed files. Notebook prints are repo-relative (`task1_llm/...`), never `/Users/...`.

## Datasets (Google Drive)

Do **not** push raw datasets to GitHub. Zip each `data/` folder, upload to Drive, and paste links below with **Anyone with the link → Viewer**.

| Dataset | Local folder (not in git) | Drive link |
|---|---|---|
| TinyStories | `task1_llm/data/` | _add_ |
| Yelp polarity | `task2_sentiment/data/` | _add_ |
| Monet / Photo (Kaggle) | `task3_gan/data/monet_jpg/`, `photo_jpg/` | _add_ |

Or one shared Drive folder: _add_

## Tasks

| Folder | Part | Dataset |
|---|---|---|
| `task1_llm/` | GPT from scratch | TinyStories |
| `task2_sentiment/` | Yelp sentiment (3 models) | Yelp polarity |
| `task3_gan/` | CycleGAN + Kaggle (structure ready; code next) | Monet / Photo |

Each task folder:

```
sneha_singh/
  src/                 executed .ipynb
  data_processed/      my preprocessing only
  checkpoints/         .pt / .pth weights
  outputs/             plots, samples, predictions
  metrics_report.csv
  failure_analysis.md
  results.md
```

## Reproduce (smoke test)

**Part 1 LLM** — `task1_llm/sneha_singh/src/config.json` → `"smoke": true`:

```bash
jupyter nbconvert --to notebook --execute --inplace task1_llm/sneha_singh/src/part1_llm.ipynb
```

**Part 2 Yelp** — kernel **Python (HW1)** (`hw1-data266`), config `"smoke": true`:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=hw1-data266 \
  task2_sentiment/sneha_singh/src/part2_sentiment.ipynb
```

## Reproducibility

- Raw logs: `reproducibility/raw_logs/sneha_singh/`
- Manifests: `reproducibility/manifests/`
- Combined report: `report/DATA266_Lab1_Report_Team_[N].pdf`

## GPU lab

Smoke-test on this laptop first. Copy checkpoints and logs off the lab machine before the slot ends.
