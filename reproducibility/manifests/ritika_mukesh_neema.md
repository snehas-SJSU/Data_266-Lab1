# Environment manifest — Ritika Mukesh Neema

Raw logs under `reproducibility/raw_logs/ritika_mukesh_neema/` are unedited.

---
## Run — Part 1 LLM (full)

- Date: 2026-09-28
- Task: `task1_llm`
- Smoke: false
- Seed: 6638 (personal, derived from student ID)
- Data: full TinyStoriesV2-GPT4 corpus — train chars: 1,824,825,663 / val chars: 22,493,387
- Architecture: hand-rolled GPT (causal self-attention, LayerNorm, GELU FFN), n_layer=4, n_head=4, n_embd=128, block_size=256, dropout=0.1
- Epochs: 12, batch_size=64, lr=3e-4 (AdamW, warmup 5% -> cosine decay)
- Checkpoint: `task1_llm/ritika_mukesh_neema/checkpoints/ckpt_final.pt` (on git; also ckpt_epoch0-11.pt)
- Metrics CSV: `task1_llm/ritika_mukesh_neema/metrics_report.csv`
- Val CE 0.7775, perplexity 2.176, bits-per-char 1.122, top-1 next-char accuracy 0.7545
- Results / failures: `task1_llm/ritika_mukesh_neema/results.md`, `failure_analysis.md`
- Raw log: `reproducibility/raw_logs/ritika_mukesh_neema/task1_llm/train_full.log`
- Device: CUDA (Google Colab, Tesla T4)
- Train time: 1726.6 s (~28.8 min); peak_memory_mb: 1749; nan_count: 0; params: 882,688
- Command: Run All on `task1_llm/ritika_mukesh_neema/src/part1_llm.ipynb` (`config.json` smoke=false)

---
## Run — Part 3 CycleGAN v2 (full — submitted)

- Date: 2026-10-04
- Task: `task3_gan`
- Smoke: false (smoke test `python run_all.py --smoke_test` passed on Colab before the run)
- Seed: 6638
- Code: `task3_gan/ritika_mukesh_neema/src/` (settings in `src/config.json`)
- Train: CUDA + AMP, 125 epochs (n_epochs=50 + n_epochs_decay=75), steps_per_epoch=800, batch_size=4, img_size=256, ngf=64, n_blocks=9, upsample=nearest, lambda_cycle=10, identity weight 5, DiffAugment color/translation/cutout, real label 0.9, EMA 0.999, no gradient clipping, image pool 50
- Train time: 16,310 s (4.53 h); 24.5 images/s; peak GPU memory 17,368 MB; nan_count: 0; params: 28,285,832
- Device: Google Colab, NVIDIA A100-SXM4 80 GB
- Command: `python train.py --ckpt_dir /content/drive/MyDrive/ritika_v2/checkpoints --out_dir /content/drive/MyDrive/ritika_v2/outputs`
- Checkpoint selection: last 30 epochs scored with the TA notebook code (raw + EMA, 60 candidates, `outputs/checkpoint_scores.csv`); submitted `checkpoints/best.pt` = epoch 110 raw. Checkpoints on Google Drive (see `task3_gan/ritika_mukesh_neema/checkpoints/README.md`)
- Generation / scoring / metrics: NVIDIA GeForce RTX 4060 Laptop GPU, Windows 11, Python 3.13, PyTorch 2.11 (cu128)
  - `python generate.py --ckpt ../checkpoints/best.pt` -> `outputs/pred_A2B` (300, 6-view average), `outputs/pred_B2A` (7,038)
  - `python evaluate_local.py` (TA script) -> `submission.csv`: FID 95.6037, MiFID 0.4054, score 48.0046 (B2A FID 93.892 / MiFID 0.3991; A2B FID 97.316 / MiFID 0.4118)
  - `python full_metrics.py --ckpt ../checkpoints/best.pt` -> `full_metrics_report.csv`, `src/full_metrics.json`
- Results / failures: `task3_gan/ritika_mukesh_neema/results.md`, `failure_analysis.md`; notebook `src/part3_cyclegan.ipynb` (built by `src/build_notebook.py` from the real files)
- Raw logs: `reproducibility/raw_logs/ritika_mukesh_neema/task3_gan/v2/` (`train_log.jsonl` per-step training log, `checkpoint_scores.csv`, console logs of generate / TA script / metrics)

---
## Run — Part 3 CycleGAN v1 (full — superseded by v2)

- Date: 2026-09-28, ~12:30-15:30 PT
- Task: `task3_gan`
- Smoke: false
- Train: CUDA, 80 epochs (n_epochs=40 + n_epochs_decay=40, best-effort split reconstruction — not logged exactly), batch_size=4, img_size=256, n_blocks=9, lambda_cycle=10, lambda_identity=5
- Train time: 1659.97 s (~27.7 min); peak_memory_mb: 1645.3; nan_count: 0; params: 28,285,832
- Checkpoint: `task3_gan/ritika_mukesh_neema/checkpoints/ckpt_final.pt` (retrieved from the Windows GPU Lab machine 2026-09-30). File is 107.9 MB, exceeding GitHub's 100MB limit, so it is excluded from this repo per .gitignore (task3_gan/**/checkpoints/*.pt) -- consistent with this team's convention of keeping oversized files on Google Drive. Available at: https://drive.google.com/file/d/1Wlu22EKbp4QATijRbNBbDOFFGHLHUiNj/view . Verified on retrieval: total parameters 28,285,832 (matches this manifest); state-dict keys G_A2B/G_B2A/D_A/D_B match this member's own naming convention.
- Kaggle-scored direction: photo->monet (B2A). Local FID 123.70, KID 0.0267 (B2A); FID 120.72, KID 0.0401 (A2B). MiFID not computed by `evaluate_local.py`.
- Cycle-consistency: A(monet->photo->monet) L1 0.109, LPIPS 0.405, content-cosine 0.873; B(photo->monet->photo) L1 0.126, LPIPS 0.337, content-cosine 0.775
- Metrics CSV: `task3_gan/ritika_mukesh_neema/outputs/history_v1/full_metrics_report.csv`
- Submission (v1): `task3_gan/ritika_mukesh_neema/outputs/history_v1/v1_submission.csv`; all v1 output files are in `outputs/history_v1/`
- Results / failures: `task3_gan/ritika_mukesh_neema/results.md`, `failure_analysis.md`
- Raw logs: `reproducibility/raw_logs/ritika_mukesh_neema/task3_gan/console_train.log`, `console_eval.log`, `console_generate.log`
- Device: CUDA (Windows GPU Lab machine, RTX 5090)
- Command: Run All on `task3_gan/ritika_mukesh_neema/src/part3_cyclegan.ipynb` (`config.json` smoke=false)
- Note: `outputs/pred_A2B/`, `pred_B2A/`, `human_audit/` sample-image subfolders not yet populated in this repo (images excluded per `.gitignore` on the source machine)

---
## Run — Part 2 Sentiment (full)

- Date: 2026-09-29
- Task: `task2_sentiment`
- Smoke: false
- Seed: 42
- Data: Hugging Face `fancyzhx/yelp_polarity` (560,000 train / 38,000 test rows), downloaded at runtime, not hosted in this repo. Split: 503,945 train / 55,993 val / 37,997 test.
- Preprocessing: lowercasing, HTML/URL stripping, punctuation removal, stopword removal (sklearn ENGLISH_STOP_WORDS), Porter stemming. Vocab size 30,000 (min_freq=2, train split only), max_len=200.
- Models: BaselineMeanEmbed, BiLSTMClassifier, TextCNNClassifier (Kim 2014, kernels 3/4/5) — all from-scratch embeddings, no pretrained word2vec/GloVe/transformer.
- Epochs: 5, batch_size=64, lr=1e-3, embed_dim=100, n_boot=1000 (bootstrap CIs)
- Checkpoints: `task2_sentiment/ritika_mukesh_neema/checkpoints/{baseline,bilstm,textcnn}.pt` (on git)
- Metrics CSV: `task2_sentiment/ritika_mukesh_neema/metrics_report.csv`
- Test accuracy / macro-F1 / ROC-AUC: baseline 0.9217 / 0.9217 / 0.9741; bilstm 0.9337 / 0.9337 / 0.9818; textcnn 0.9260 / 0.9260 / 0.9791
- Train time: baseline 124.9s, bilstm 797.5s, textcnn 267.2s; peak_memory_mb: 2123 / 2274 / 2450; params: 3,000,101 / 3,235,777 / 3,120,601
- McNemar (baseline vs. each): vs. bilstm b=817, c=1275, stat=99.832, p<0.001; vs. textcnn b=903, c=1068, stat=13.646, p=0.00022
- Results / failures: `task2_sentiment/ritika_mukesh_neema/results.md`, `failure_analysis.md`
- Raw log: `reproducibility/raw_logs/ritika_mukesh_neema/task2_sentiment/console_run_all_full.log`
- Device: CUDA (Google Colab, Tesla T4)
- Command: `cd src && python run_all.py --train_csv <yelp train parquet> --test_csv <yelp test parquet>` (`config.json` smoke=false)
