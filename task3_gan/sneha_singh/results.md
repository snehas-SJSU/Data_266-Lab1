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
- nearest upsample + stride-1 conv (no ConvTranspose)
- label smoothing 0.9
- ~800 photos resampled each epoch; the shorter Monet loader is cycled so that photo subset is used
- full photo set used for A2B inference

## Hardware
- CUDA GPU, AMP, `smoke=false`
- 80 epochs, batch size 4
- `train_time_sec` ≈ 15341 (~4.3 h); peak memory ~6568 MB
- The saved log prints `device=cuda`. It does not record the GPU name.
- Smoke run earlier on Mac (MPS)
- Raw logs: `reproducibility/raw_logs/sneha_singh/task3_gan/train_smoke.log`, `train_full.log`

## Metrics (local — already run)

From `evaluate_local.py` on Mac, on this retrain:

| Direction | FID | KID | MiFID | LPIPS | content cos | cycle L1 |
|---|---|---|---|---|---|---|
| A2B (photo→Monet) | 89.99 | 0.018 | 0.403 | 0.452 | 0.532 | 0.112 |
| B2A (Monet→photo) | 92.23 | 0.030 | 0.423 | 0.365 | 0.586 | 0.103 |

Full A2B (7038) + full B2A (300 Monet). A2B is the Kaggle direction.

Also in CSV: precision/recall, G/D/cycle/identity losses, grad norm, nan_count=0, params, train time, peak memory.

Same A2B FID/MiFID are in `submission.csv`.

## Kaggle

- Kaggle team name: PairProgramming_Team_5
- Upload file: `submission.csv` (`ID,FID,MiFID` = 1, 89.99, 0.403)
- Public score: _fill after Kaggle shows it_
- Private score: _fill after Kaggle shows it_
- Rank: _fill after Kaggle shows it_

## Notes
- LSGAN + cycle + identity matches the CycleGAN paper recipe for this dataset.
- Batch 4 + image buffer helped on the small Monet set (~300).
- We keep the best-cycle checkpoint, not just the last epoch.
