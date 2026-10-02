# Part 3 — CycleGAN photo ↔ Monet (Sneha Singh)

Dataset: unpaired Monet paintings + photos (Kaggle). Own CycleGAN only — no pretrained image models on outputs.

## Architecture
- Generators: ResNet (reflection pad, instance norm, 9 residual blocks @256), tanh output
- Discriminators: PatchGAN (LSGAN / MSE)
- Losses: adversarial + cycle-consistency (λ=10) + identity (0.5×λ)
- Image buffer (pool) for discriminator updates
- Upsample: nearest-neighbor ×2, then a stride-1 conv
- Checkpoint: last epoch

## This checkpoint
The numbers below are the 80-epoch Colab run in `Downloads/part3_done.zip` (40 constant + 40 decay), batch 4, about 800 photos resampled each epoch. That is the best professor score. Later Colab runs did not beat it.

## Hardware
- Colab Tesla T4, AMP, `smoke=false`
- 80 epochs, batch size 4
- `train_time_sec` ≈ 15341 (~4.3 h); peak memory ~6568 MB
- Raw logs: `reproducibility/raw_logs/sneha_singh/task3_gan/train_smoke.log`, `train_full.log`

## Folders
Professor names:

- `outputs/pred_A2B` — Monet → photo (300 images), scored against real photos
- `outputs/pred_B2A` — photo → Monet (7,038 images), scored against real Monet

## Metrics
From `evaluate_local.py`, using the professor script: Inception-v3, first 300 sorted images, MiFID = mean cosine distance. FID and MiFID in the table are that script. KID, LPIPS, and cycle L1 are the same images.

| Direction | FID | KID | MiFID | LPIPS | content cos | cycle L1 |
|---|---|---|---|---|---|---|
| A2B (Monet→photo) | 109.33 | 0.030 | 0.425 | 0.365 | 0.586 | 0.103 |
| B2A (photo→Monet) | 105.06 | 0.018 | 0.405 | 0.452 | 0.532 | 0.112 |

`submission.csv` is the average of both directions: FID **107.20**, MiFID **0.415**. These numbers were recalculated from `part3_done.zip` with the professor script. Later Colab runs scored 110.18, 117.49, and 119.69, all worse, so they are not the submission.

Also in `full_metrics_report.csv`: precision/recall, G/D/cycle/identity losses, grad norm, nan_count=0, params, train time, peak memory.

## Kaggle
- Team: PairProgramming_Team_5
- Upload file: `submission.csv` (`ID,FID,MiFID` = 1, 107.1960865081875, 0.41492947936058044)
- Score = (FID + MiFID) / 2 = (107.196 + 0.415) / 2 ≈ 53.81
- Public score: ≈53.81 (_fill exact Kaggle value_)
- Private score: _fill after Kaggle shows it_
- Rank: _fill after Kaggle shows it_
