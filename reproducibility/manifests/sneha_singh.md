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
- Note: full GPU run (100K/10K, ≥10 epochs) not done yet — will add a new section after that run.

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

## Run — Part 2 Sentiment (full)

- Date: 2026-09-19 02:37
- Task: `task2_sentiment`
- Smoke: false
- Train/val/test: 15000 / 3000 / 3000 · Epochs: 5
- Checkpoints (on git):
  - `task2_sentiment/sneha_singh/checkpoints/baseline_full.pt`
  - `task2_sentiment/sneha_singh/checkpoints/experimental_a_full.pt`
  - `task2_sentiment/sneha_singh/checkpoints/experimental_b_full.pt`
- Raw logs:
  - `reproducibility/raw_logs/sneha_singh/task2_sentiment/baseline_full.log`
  - `reproducibility/raw_logs/sneha_singh/task2_sentiment/experimental_a_full.log`
  - `reproducibility/raw_logs/sneha_singh/task2_sentiment/experimental_b_full.log`
- Metrics CSV: `task2_sentiment/sneha_singh/metrics_report.csv`
- Results / failures: `results.md`, `failure_analysis.md`
- Python: 3.12.13 · PyTorch: 2.13.0 · Device: mps (Apple Silicon)
- Platform: macOS-26.2-arm64-arm-64bit
- Command: Run All on `task2_sentiment/sneha_singh/src/part2_sentiment.ipynb` (smoke=false)

---

## Run — Part 3 CycleGAN (smoke)

- Date: 2026-09-24 16:52 (last smoke; earlier same-day retry omitted)
- Task: `task3_gan`
- Smoke: true
- Checkpoint: `task3_gan/sneha_singh/checkpoints/best_smoke.pt` (local only; later removed)
- Pred A2B / B2A: under `task3_gan/sneha_singh/outputs/` (smoke sizes)
- Raw log: `reproducibility/raw_logs/sneha_singh/task3_gan/train_smoke.log`
- Python: 3.12.13 · PyTorch: 2.14.0 · Device: mps
- Command: Run All on `part3_cyclegan.ipynb` (smoke=true)

---

## Run — Part 3 CycleGAN (full)

- Date: train 2026-09-24/25 (Colab); local eval 2026-09-25 (Mac, incl. full B2A)
- Task: `task3_gan`
- Smoke: false
- Train: Google Colab **Tesla T4 (15.64 GB)**, CUDA + AMP, 80 epochs (40+40), batch_size=4
- Train time: ~5238 s (~87 min); peak_memory_mb: ~4820
- Checkpoint: `task3_gan/sneha_singh/checkpoints/best.pt` (**Drive** — GitHub 100 MB limit; link in root README)
- Pred A2B: `outputs/pred_A2B` (7038 JPGs)
- Pred B2A: `outputs/pred_B2A` (300 JPGs, full Monet set)
- Metrics CSV: `task3_gan/sneha_singh/full_metrics_report.csv`
- Submission: `task3_gan/sneha_singh/submission.csv` (fid≈95.13, mifid≈0.411) — Kaggle upload pending
- Local A2B: FID 95.13, MiFID 0.411 · Local B2A: FID 96.36, MiFID 0.433
- Results / failures: `results.md`, `failure_analysis.md`
- Eval command: `python task3_gan/sneha_singh/evaluate_local.py` (Mac; fidelity backend CPU, LPIPS on MPS)
- Train command: Colab Run All on `part3_cyclegan.ipynb` (smoke=false)
- Exact GPU: Tesla T4 (`torch.cuda.get_device_name(0)`)
