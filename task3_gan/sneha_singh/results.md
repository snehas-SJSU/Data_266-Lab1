# Part 3 — CycleGAN photo ↔ Monet (Sneha Singh)

Dataset: unpaired Monet paintings + photos (Kaggle). Own CycleGAN only — no pretrained image models on outputs.

## Architecture
- Generators: ResNet (reflection pad, instance norm, 9 residual blocks @256), tanh output
- Discriminators: PatchGAN (LSGAN / MSE)
- Losses: adversarial + cycle-consistency (λ=10) + identity (0.5×λ)
- Image buffer (pool) for discriminator updates

## Hyperparameters
See `src/config.json`.

Full-run budget:
- batch_size=4
- 40 constant + 40 decay epochs (not 200)
- ~400 photos resampled each epoch; full photo set used for A2B inference

## Hardware
- Google Colab, **Tesla T4 (15.64 GB)**, AMP, `smoke=false`
- 80 epochs, batch size 4
- ~87 min (`train_time_sec` ≈ 5238); peak memory ~4820 MB
- Smoke run earlier on Mac (MPS)

## Metrics (local — already run)

From `evaluate_local.py` on Mac, A2B photo→Monet:

| Metric | Value |
|---|---|
| FID | 95.13 |
| MiFID | 0.411 |
| Leaderboard proxy (FID+MiFID)/2 | 47.77 |

Same numbers are in `submission.csv` and `full_metrics_report.csv`.

## Kaggle

- Team 5
- Upload file: `submission.csv` (`fid,mifid` = 95.13, 0.411)
- Public score: _fill after Kaggle shows it_
- Private score: _fill after Kaggle shows it_
- Rank: _fill after Kaggle shows it_

## Notes
- LSGAN + cycle + identity matches the CycleGAN paper recipe for this dataset.
- Batch 4 + image buffer helped on the small Monet set (~300).
- We keep the best-cycle checkpoint, not just the last epoch.
