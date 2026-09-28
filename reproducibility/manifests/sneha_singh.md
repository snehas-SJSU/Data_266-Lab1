# Environment manifest — Sneha Singh

Lab PDF (Section 5): record package/library versions, environment, and which
checkpoint maps to which result in the report. One section per reported result
(smoke and/or full). Raw training logs under `reproducibility/raw_logs/` stay
unedited.

---
## Run — Part 1 LLM (smoke)

- Date: 2026-09-19 01:57 (last successful smoke; earlier same-day retries omitted)
- Task: `task1_llm`
- Smoke: true
- Checkpoint: `task1_llm/sneha_singh/checkpoints/best_smoke.pt`
- Metrics CSV: `task1_llm/sneha_singh/metrics_report.csv`
- Results / failures: `task1_llm/sneha_singh/results.md`, `failure_analysis.md`
- Raw log: `reproducibility/raw_logs/sneha_singh/task1_llm/train_smoke.log`
- Python: 3.12.13
- PyTorch: 2.13.0
- Device: mps (Apple Silicon)
- Platform: macOS-26.2-arm64-arm-64bit
- Command: Run All on `task1_llm/sneha_singh/src/part1_llm.ipynb` (`config.json` smoke=true)
- Note: smoke metrics were replaced in `metrics_report.csv` by the full run below. This smoke log is unchanged.

---
## Run — Part 1 LLM (full)

- Date: 2026-09-28 14:12
- Task: `task1_llm`
- Smoke: false
- Split: 100000 train / 10000 val windows, 10 epochs, block 128
- Checkpoint: `task1_llm/sneha_singh/checkpoints/best.pt` (12 MB, on git)
- Metrics CSV: `task1_llm/sneha_singh/metrics_report.csv`
- Val CE 0.8571, perplexity 2.356, top-1 next-char accuracy 0.729
- Results / failures: `task1_llm/sneha_singh/results.md`, `failure_analysis.md`
- Raw log: `reproducibility/raw_logs/sneha_singh/task1_llm/train_full.log`
- Python: 3.11.16
- PyTorch: 2.12.0.dev20260408+cu128
- Device: cuda: NVIDIA GeForce RTX 5090
- Platform: Windows-10-10.0.26200-SP0
- Train time: 457 s; peak_memory_mb: 349; nan_count: 0; params: 3254272
- Command: Run All on `task1_llm/sneha_singh/src/part1_llm.ipynb` (`config.json` smoke=false)
- Dataset file `TinyStories-train.txt` is not on git (Drive).

---

## Run — Part 2 Sentiment (smoke)

- Date: 2026-09-19 02:28
- Task: `task2_sentiment`
- Smoke: true
- Train/val/test: 2000 / 500 / 500 · Epochs: 2
- Checkpoints:
  - `task2_sentiment/sneha_singh/checkpoints/baseline_smoke.pt`
  - `task2_sentiment/sneha_singh/checkpoints/experimental_a_smoke.pt`
  - `task2_sentiment/sneha_singh/checkpoints/experimental_b_smoke.pt`
- Raw logs:
  - `reproducibility/raw_logs/sneha_singh/task2_sentiment/baseline_smoke.log`
  - `reproducibility/raw_logs/sneha_singh/task2_sentiment/experimental_a_smoke.log`
  - `reproducibility/raw_logs/sneha_singh/task2_sentiment/experimental_b_smoke.log`
- Metrics CSV: `task2_sentiment/sneha_singh/metrics_report.csv` (overwritten by full run below)
- Python: 3.12.13 · PyTorch: 2.13.0 · Device: mps
- Command: Run All on `part2_sentiment.ipynb` (smoke=true)

---

## Run — Part 3 CycleGAN (smoke)

- Date: 2026-09-24 16:52 (last smoke; earlier same-day retry omitted)
- Task: `task3_gan`
- Smoke: true
- Checkpoint: `task3_gan/sneha_singh/checkpoints/best_smoke.pt` (local only; later removed)
- Pred A2B / B2A: under `task3_gan/sneha_singh/outputs/` (smoke sizes)
- Raw log: `reproducibility/raw_logs/sneha_singh/task3_gan/train_smoke.log` (2 epoch lines; recovered from notebook smoke stdout)
- Python: 3.12.13 · PyTorch: 2.14.0 · Device: mps
- Command: Run All on `part3_cyclegan.ipynb` (smoke=true)

---

## Run — Part 3 CycleGAN (full)

- Date: train and local eval 2026-09-28
- Task: `task3_gan`
- Smoke: false
- Train: CUDA + AMP, 80 epochs (40+40), batch_size=4, nearest upsample, label_smoothing 0.9, ~800 photos/epoch
- Train time: 15341 s (~4.3 h); peak_memory_mb: 6568
- GPU name: not printed in `train_full.log` (log line is `device=cuda` only)
- Checkpoint: `task3_gan/sneha_singh/checkpoints/best.pt` (**Drive** — GitHub 100 MB limit; link in root README)
- Pred A2B: `outputs/pred_A2B` (7038 JPGs)
- Pred B2A: `outputs/pred_B2A` (300 JPGs, full Monet set)
- Raw log: `reproducibility/raw_logs/sneha_singh/task3_gan/train_full.log` (80 epoch lines)
- Metrics CSV: `task3_gan/sneha_singh/full_metrics_report.csv`
- Submission: `task3_gan/sneha_singh/submission.csv` (FID 89.99, MiFID 0.403) — Kaggle upload pending
- Local A2B: FID 89.99, MiFID 0.403 · Local B2A: FID 92.23, MiFID 0.423
- Results / failures: `results.md`, `failure_analysis.md`
- Eval command: `python task3_gan/sneha_singh/evaluate_local.py` (Mac; fidelity backend CPU, LPIPS on MPS)
- Train command: Run All on `part3_cyclegan.ipynb` (smoke=false)

## Run — Part 2 Sentiment (full)
- Date: 2026-09-26 23:36
- Task: task2_sentiment
- Smoke: False
- Train/val/test: 99989 / 10000 / 10000 (100000 sampled; 11 empty reviews dropped)
- Test macro-F1: baseline 0.9226, BiLSTM 0.9184, TextCNN 0.9207
- Epochs: 5
- Checkpoint (baseline): task2_sentiment/sneha_singh/checkpoints/baseline_full.pt
- Checkpoint (experimental_a): task2_sentiment/sneha_singh/checkpoints/experimental_a_full.pt
- Checkpoint (experimental_b): task2_sentiment/sneha_singh/checkpoints/experimental_b_full.pt
- Raw log (baseline): reproducibility/raw_logs/sneha_singh/task2_sentiment/baseline_full.log
- Raw log (experimental_a): reproducibility/raw_logs/sneha_singh/task2_sentiment/experimental_a_full.log
- Raw log (experimental_b): reproducibility/raw_logs/sneha_singh/task2_sentiment/experimental_b_full.log
- Metrics CSV: task2_sentiment/sneha_singh/metrics_report.csv
- Results: task2_sentiment/sneha_singh/results.md
- Python: 3.12.13
- PyTorch: 2.14.0
- Device: Apple M4 (10 cores: 4 performance, 6 efficiency), 16 GB unified, MPS
- Platform: macOS-26.2-arm64-arm-64bit
- Command: Run All on task2_sentiment/sneha_singh/src/part2_sentiment.ipynb (config.json smoke=false)
