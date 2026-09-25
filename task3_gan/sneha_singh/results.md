# Part 3 — CycleGAN photo ↔ Monet (Sneha Singh)

Dataset: unpaired Monet paintings + photos (Kaggle). Own CycleGAN only — no pretrained image models on outputs.

## Architecture
- Generators: ResNet (reflection pad, instance norm, 9 residual blocks @256), tanh output
- Discriminators: PatchGAN (LSGAN / MSE)
- Losses: adversarial + cycle-consistency (λ=10) + identity (0.5×λ)
- Image buffer (pool) for discriminator updates

## Hyperparameters
See `src/config.json`.

Full-run budget (so a GPU session finishes):
- batch_size=4
- 40 constant + 40 decay epochs (not 200)
- ~400 photos resampled each epoch (full photo set still used for A2B inference → metrics)

## Hardware
- Full train on **Google Colab**, GPU = **Tesla T4 (15.64 GB)**, AMP on, `smoke=false`
- 80 epochs (40 + 40), batch size 4
- About **87 min** wall time (`train_time_sec` ≈ 5238); peak memory ~4820 MB
- Earlier smoke run was on my Mac (MPS)

## Metrics
I ran `evaluate_local.py` on my Mac after downloading Colab outputs (A2B = photo→Monet):
- FID **95.13**, MiFID **0.411**, average proxy **47.77**
- More columns (KID, LPIPS, etc.) are in `full_metrics_report.csv`
- Kaggle file is `submission.csv` (same FID/MiFID)

## Kaggle
- Team 5
- Upload **`submission.csv`** (FID + MiFID only — not `images.zip`)
- Public / private score: _fill after I submit_
- Rank: _fill after I submit_

## Notes
Why this config (for viva):
- LSGAN + cycle + identity is the standard stable CycleGAN recipe for Monet↔photo.
- Small batch + image buffer reduces oscillation on a tiny Monet set (~300 images).
- Checkpoint on best epoch, not last — GAN losses are not monotonic.
