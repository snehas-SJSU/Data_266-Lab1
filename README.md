# DATA266 Lab 1 — Team 5

**GitHub:** https://github.com/snehas-SJSU/Data_266-Lab1  
**Team:** 5  

| Member | Folder name |
|---|---|
| Sneha Singh | `sneha_singh/` |
| Ritika Mukesh Neema | `ritika_mukesh_neema/` |

---

## Setup

Python 3.11 or 3.12 (our runs used 3.12). A CUDA GPU is needed to train Part 3. Parts 1 and 2 also run on CPU or Apple MPS.

```bash
git clone https://github.com/snehas-SJSU/Data_266-Lab1.git
cd Data_266-Lab1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Large files (not in git): download and place them like this before running.**

| File | Download | Put it here |
|---|---|---|
| `TinyStories-train.txt` (Sneha, Part 1 full run) | Drive link 1 below (the notebook also downloads it if missing) | `task1_llm/data/` |
| TinyStoriesV2-GPT4 train / valid (Ritika, Part 1) | Hugging Face `roneneldan/TinyStories` | any folder; pass the paths to `run_all.py` |
| Yelp polarity (Part 2) | Sneha's notebook downloads it. Ritika: Hugging Face `fancyzhx/yelp_polarity` parquet files | Ritika: pass the paths to `run_all.py` |
| `monet_jpg/` (300 JPGs), `photo_jpg/` (7,038 JPGs) | Drive links 2 and 3 below | `task3_gan/data/monet_jpg/`, `task3_gan/data/photo_jpg/` (no extra parent folder) |
| Ritika Part 3 checkpoint | Ritika's Drive link below | `task3_gan/ritika_mukesh_neema/checkpoints/ckpt_final.pt` |

**Viewing results without retraining:** each notebook is saved with its outputs, so open it to see them. Running all cells starts a new full training run (Part 3 takes hours on a GPU). Metrics, samples, and plots are also in each member's `results.md`, metrics CSV, and `outputs/`.

---

## Smoke test (one command)

```bash
bash run_smoke.sh
```

Team smoke test (Sneha Singh, Ritika Mukesh Neema). Runs the Part 1 GPT with `smoke: true` (TinyStories-valid, 256 train / 64 val, 2 epochs, about 1 minute on CPU). The notebook downloads the small TinyStories-valid file itself. The script works in a temporary copy of the repo, so committed results, raw logs, and manifests are not touched. It prints the output folder when it finishes.

---

## How to run — all members

Run every command from the repo root after Setup. Each `config.json` is set to the full run (`"smoke": false`). Set `"smoke": true` for a short run.

| Part | Member | Command | Data |
|---|---|---|---|
| 1 — LLM | Sneha | `jupyter nbconvert --to notebook --execute --inplace task1_llm/sneha_singh/src/part1_llm.ipynb` | Notebook downloads TinyStories |
| 1 — LLM | Ritika | `cd task1_llm/ritika_mukesh_neema/src && python run_all.py --train_txt <TinyStoriesV2-GPT4-train.txt> --val_txt <TinyStoriesV2-GPT4-valid.txt>` | TinyStoriesV2-GPT4 train / valid text files (Hugging Face) |
| 2 — Sentiment | Sneha | `jupyter nbconvert --to notebook --execute --inplace task2_sentiment/sneha_singh/src/part2_sentiment.ipynb` | Notebook downloads Yelp polarity |
| 2 — Sentiment | Ritika | `cd task2_sentiment/ritika_mukesh_neema/src && python run_all.py --train_csv <train.parquet> --test_csv <test.parquet>` | Yelp polarity train / test parquet files (Hugging Face `fancyzhx/yelp_polarity`) |
| 3 — CycleGAN | Sneha | `jupyter nbconvert --to notebook --execute --inplace task3_gan/sneha_singh/src/part3_cyclegan.ipynb` | `monet_jpg/`, `photo_jpg/` from Drive into `task3_gan/data/` |
| 3 — CycleGAN | Ritika | `cd task3_gan/ritika_mukesh_neema/src && python run_all.py` | Same as above |

Part 3 — rebuild the submitted outputs from `best.pt`, then the evaluation script (both directions averaged, writes `submission.csv`):

```bash
python task3_gan/sneha_singh/make_submission.py
python task3_gan/sneha_singh/evaluate_local.py
```

Part 3 trains on a CUDA GPU. Parts 1 and 2 also run on CPU or Apple MPS, only slower. Ritika's `run_all.py` scripts also accept `--smoke_test`.

### Checkpoint → result

| Part | Member | Checkpoint | Result |
|---|---|---|---|
| 1 | Sneha | `task1_llm/sneha_singh/checkpoints/best.pt` (epoch 10) | Val CE 0.857, perplexity 2.36 |
| 1 | Ritika | `task1_llm/ritika_mukesh_neema/checkpoints/ckpt_final.pt` (12 epochs) | Val CE 0.778, perplexity 2.18 |
| 2 | Sneha | `task2_sentiment/sneha_singh/checkpoints/baseline_full.pt`, `experimental_a_full.pt` (BiLSTM), `experimental_b_full.pt` (TextCNN) | Best: BiLSTM, macro-F1 0.942 |
| 2 | Ritika | `task2_sentiment/ritika_mukesh_neema/checkpoints/baseline.pt`, `bilstm.pt`, `textcnn.pt` | Best: BiLSTM, macro-F1 0.934 |
| 3 | Sneha | `task3_gan/sneha_singh/checkpoints/best.pt` (epoch 93) | `submission.csv`: FID 99.84, MiFID 0.412 (score 50.13) |
| 3 | Ritika | `task3_gan/ritika_mukesh_neema/checkpoints/ckpt_final.pt` (Drive, epoch 80) | Photo→Monet FID 123.70, Monet→photo FID 120.72 |

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

- `src/part1_llm.ipynb`, `src/config.json`, scripts `data.py`, `model.py`, `train.py`, `generate.py`, `metrics.py`, `run_all.py`
- `metrics_report.csv`, `results.md`, `failure_analysis.md`
- `checkpoints/ckpt_epoch0.pt` … `ckpt_epoch11.pt`, `ckpt_final.pt`
- Loss curve, generated samples, generation metrics under `outputs/`
- Raw log: `reproducibility/raw_logs/ritika_mukesh_neema/task1_llm/train_full.log`

**Model:** 4 layers, 4 heads, embedding 128, block size 256, 882,688 parameters. 12 epochs, batch 64, lr 3e-4 with warmup + cosine decay, seed 6638.

**Status:** full run done (Colab Tesla T4, about 28.8 min). Val CE 0.778, perplexity 2.18, top-1 next-char accuracy 0.755.

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
# "smoke": false in config.json
jupyter nbconvert --to notebook --execute --inplace \
  task2_sentiment/sneha_singh/src/part2_sentiment.ipynb
```

**Status:** full run done (549,953 train / 10k val / 38k test, 5 epochs, Colab Tesla T4). Best model is the BiLSTM, macro-F1 0.942.

### Ritika Mukesh Neema — `task2_sentiment/ritika_mukesh_neema/`

- `src/part2_sentiment.ipynb`, `src/config.json`, scripts `preprocess.py`, `models.py`, `train.py`, `metrics.py`, `error_analysis.py`, `run_all.py`
- 3 models: baseline (mean pool), BiLSTM, TextCNN
- `metrics_report.csv`, `results.md`, `failure_analysis.md` (20 errors)
- Checkpoints: `checkpoints/baseline.pt`, `bilstm.pt`, `textcnn.pt`
- Full metrics, McNemar results, test predictions under `outputs/`
- Raw log: `reproducibility/raw_logs/ritika_mukesh_neema/task2_sentiment/console_run_all_full.log`

**Status:** full run done (503,945 train / 55,993 val / 37,997 test, 5 epochs, Colab Tesla T4). Best model is the BiLSTM, macro-F1 0.934.

---

## Part 3 — CycleGAN (photo ↔ Monet)

**Shared data:** `task3_gan/data/monet_jpg/`, `photo_jpg/` (class Kaggle). Drive link in Common table below.

### Sneha Singh — `task3_gan/sneha_singh/`

- `src/part3_cyclegan.ipynb` (Colab run with outputs), `src/config.json`, `make_submission.py`, `evaluate_local.py`, `score_checkpoints.py`, `audit_agreement.py`
- `submission.csv` (FID + MiFID for Kaggle)
- `full_metrics_report.csv`, `metrics_report.csv` (long format: final metrics, human audit, per-epoch losses, scored checkpoints), `results.md`, `failure_analysis.md`
- Sample grid + loss curves; checkpoint scores `outputs/checkpoint_scores_epochs86_100.csv`; 30-image audit under `outputs/human_audit/`
- `checkpoints/best.pt` (43 MB, in git): epoch 93

**How to run**

- Train: CUDA GPU, `"smoke": false`, AMP (100 epochs, ~5.4 h on a Colab L4, peak ~3361 MB).
- Submission from `best.pt`, then metrics:

```bash
python task3_gan/sneha_singh/make_submission.py
python task3_gan/sneha_singh/evaluate_local.py
```

**Kaggle (Sneha)**

| Item | Value |
|---|---|
| Upload file | `task3_gan/sneha_singh/submission.csv` |
| FID / MiFID | 99.84 / 0.412 (average of both directions, professor script on Colab L4) |
| Score | 50.13 |
| Leaderboard rank | to be recorded after the Kaggle upload |

**Status:** `best.pt` is epoch 93 of the 100-epoch run, selected by professor-script score among epochs 86–100. Professor eval is in `submission.csv` (FID 99.84, MiFID 0.412). A2B Monet→photo FID 101.39 (6-view averaging). B2A photo→Monet FID 98.29.

### Ritika Mukesh Neema — `task3_gan/ritika_mukesh_neema/`

- `src/part3_cyclegan.ipynb`, `src/config.json`, scripts `dataset.py`, `models.py`, `train.py`, `generate.py`, `evaluate_local.py`, `kaggle_score.py`, `human_audit.py`, `run_all.py`
- `submission.csv`, `metrics_report.csv`, `outputs/full_metrics_report.csv`, `results.md`, `failure_analysis.md`
- Loss curves, train logs, generation manifest under `outputs/`
- `best.pt` on Drive (link below)
- Raw logs: `reproducibility/raw_logs/ritika_mukesh_neema/task3_gan/`

**Model:** ResNet generators (9 blocks), PatchGAN discriminators, batch 1, 80 epochs (40 + 40 decay), lr 2e-4, cycle λ 10. 28,285,832 parameters. NVIDIA RTX 5090, about 27.7 min.

**Status:** photo→Monet (B2A) FID 123.70, Monet→photo (A2B) FID 120.72 (her `evaluate_local.py`, 300 images).

| Item | Value |
|---|---|
| Upload file | `task3_gan/ritika_mukesh_neema/submission.csv` |
| Local FID (training-split, not official) | 123.70 |
| Kaggle FID / MiFID | 104.30 / 0.413 (her `kaggle_score.py`: all 7,038 photo→Monet images vs `real_stats.npz`) |
| Public score | -52.3574 (own submission, own Kaggle account) |

---

## Google Drive

### 1. Common links — 3 shared datasets

Same Drive folder for the team (raw data only):

| # | Dataset | Link |
|---|---|---|
| 1 | TinyStories-train.txt (Part 1). On Drive because the file is 1.79 GB, over GitHub’s 100 MB limit. | https://drive.google.com/drive/folders/12MFVOo6T3QRW3X6THiDkh5svtNgL13W_?usp=share_link |
| 2 | Monet paintings — `monet_jpg` (Part 3) | https://drive.google.com/drive/folders/1BXYfhW8uZ6umK1TZFW8Un62mZVK72L7Y?usp=share_link |
| 3 | Photos — `photo_jpg` (Part 3, same Drive folder as Monet) | https://drive.google.com/drive/folders/1BXYfhW8uZ6umK1TZFW8Un62mZVK72L7Y?usp=share_link |

Part 2 is not in this table. Yelp polarity is not a file we host. Each notebook downloads `fancyzhx/yelp_polarity` from Hugging Face at run time.

### 2. Individual links — Sneha / Ritika

Part 1 `best.pt` (12 MB), Part 2 checkpoints, and Sneha's Part 3 `best.pt` (43 MB) are **on git**. The datasets stay in the common table above.

**Sneha Singh**

| File | Link |
|---|---|
| Part 3 `best.pt` (backup copy; also in git) | https://drive.google.com/drive/folders/12AgM95RbZAQUyouH7sukM9nu_rVTD3C9?usp=share_link |

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
