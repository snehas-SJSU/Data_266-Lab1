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

From `evaluate_local.py` on Mac:

| Direction | FID | KID | MiFID | LPIPS | content cos | cycle L1 |
|---|---|---|---|---|---|---|
| A2B (photo→Monet) | 95.13 | 0.021 | 0.411 | 0.436 | 0.576 | 0.117 |
| B2A (Monet→photo) | 96.36 | 0.034 | 0.433 | 0.329 | 0.632 | 0.102 |

Full A2B (7038) + full B2A (300 Monet). A2B is the Kaggle direction.

Also in CSV: precision/recall, G/D/cycle/identity losses, grad norm, nan_count=0, params, train time, peak memory.

Same A2B FID/MiFID are in `submission.csv`.

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
