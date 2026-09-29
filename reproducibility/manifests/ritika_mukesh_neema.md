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
## Run — Part 3 CycleGAN (full)

- Date: 2026-09-28, ~12:30-15:30 PT
- Task: `task3_gan`
- Smoke: false
- Train: CUDA, 80 epochs (n_epochs=40 + n_epochs_decay=40, best-effort split reconstruction — not logged exactly), batch_size=1, img_size=256, n_blocks=9, lambda_cycle=10, lambda_identity=5
- Train time: 1659.97 s (~27.7 min); peak_memory_mb: 1645.3; nan_count: 0; params: 28,285,832
- Checkpoint: `ckpt_final.pt` — trained and used for the Kaggle submission, but stored only on the Windows GPU Lab machine used for training; not yet uploaded to git or Drive. Pending upload.
- Kaggle-scored direction: photo->monet (B2A). Local FID 123.70, KID 0.0267 (B2A); FID 120.72, KID 0.0401 (A2B). MiFID not computed by `evaluate_local.py`.
- Cycle-consistency: A(monet->photo->monet) L1 0.109, LPIPS 0.405, content-cosine 0.873; B(photo->monet->photo) L1 0.126, LPIPS 0.337, content-cosine 0.775
- Metrics CSV: `task3_gan/ritika_mukesh_neema/outputs/full_metrics_report.csv`
- Submission: `task3_gan/ritika_mukesh_neema/submission.csv` (FID 123.70 summary row); full per-image Kaggle upload preserved at `submission_kaggle_upload.csv`
- Results / failures: `task3_gan/ritika_mukesh_neema/results.md`, `failure_analysis.md`
- Raw logs: `reproducibility/raw_logs/ritika_mukesh_neema/task3_gan/console_train.log`, `console_eval.log`, `console_generate.log`
- Device: CUDA (Windows GPU Lab machine, RTX 5090)
- Command: Run All on `task3_gan/ritika_mukesh_neema/src/part3_cyclegan.ipynb` (`config.json` smoke=false)
- Note: `outputs/pred_A2B/`, `pred_B2A/`, `human_audit/` sample-image subfolders not yet populated in this repo (images excluded per `.gitignore` on the source machine)

---
## Run — Part 2 Sentiment

- Status: not yet run — planned for 2026-09-29. Blocked twice today by Colab free-tier GPU quota exhaustion during the smoke/full run attempt.
