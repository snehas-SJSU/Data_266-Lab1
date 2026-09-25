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

## Metrics
Local eval on Mac (`evaluate_local.py`), A2B photo→Monet:
- FID **95.13**, MiFID **0.411**, (FID+MiFID)/2 **47.77**
- Full table: `full_metrics_report.csv`
- Kaggle file: `submission.csv`

## Kaggle
- Team 5
- File: `submission.csv` (FID + MiFID)
- Public / private score: _fill after submit_
- Rank: _fill after submit_

## Notes
- LSGAN + cycle + identity matches the CycleGAN paper recipe for this dataset.
- Batch 4 + image buffer helped on the small Monet set (~300).
- We keep the best-cycle checkpoint, not just the last epoch.
