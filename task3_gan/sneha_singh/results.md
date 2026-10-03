# Part 3 — CycleGAN photo ↔ Monet (Sneha Singh)

Dataset: unpaired Monet paintings + photos (Kaggle). Own CycleGAN only — no pretrained image models on outputs.

## Architecture
- Generators: ResNet (reflection pad, instance norm, 9 residual blocks @256), tanh output
- Discriminators: PatchGAN (LSGAN / MSE)
- Losses: adversarial + cycle-consistency (λ=10) + identity (`lambda_identity` 5.0 × λ = weight 50)
- Image buffer (pool of 50) for discriminator updates; real labels smoothed to 0.9
- Upsample: nearest-neighbor ×2, then a stride-1 conv
- Checkpoint: last epoch (epoch 100)

## This checkpoint
The numbers below are the 100-epoch Colab run (50 constant + 50 linear decay), batch 4, 300 steps per epoch with 1,200 photos resampled each epoch. `checkpoints/best.pt` (epoch 100) is the checkpoint in `submission.csv`.

## Hardware
- Colab NVIDIA L4, AMP, `smoke=false`
- 100 epochs, batch size 4
- `train_time_sec` 14,349 (~4.0 h); peak GPU memory 6,570 MB (`torch.cuda.max_memory_allocated()` read in the same Colab session after the run)
- Raw log: `reproducibility/raw_logs/sneha_singh/task3_gan/train_full_100ep.log`. The earlier 80-epoch run's log stays in `train_full.log`.

## Folders
Professor names:

- `outputs/pred_A2B` — Monet → photo (300 images), scored against real photos
- `outputs/pred_B2A` — photo → Monet (7,038 images), scored against real Monet

## Metrics
FID and MiFID come from the unchanged professor evaluation script, run on Colab (L4) inside the notebook: Inception-v3, first 300 sorted images, MiFID = mean cosine distance. KID, precision/recall, LPIPS, and content cosine come from `evaluate_local.py` on `outputs/pred_*` (Apple MPS). Cycle L1 is from the notebook's cycle check.

| Direction | FID | KID | MiFID | Precision | Recall | LPIPS | content cos | cycle L1 |
|---|---|---|---|---|---|---|---|---|
| A2B (Monet→photo) | 108.12 | 0.026 | 0.422 | 0.610 | 0.113 | 0.274 | 0.688 | 0.089 |
| B2A (photo→Monet) | 105.91 | 0.021 | 0.412 | 0.392 | 0.410 | 0.279 | 0.717 | 0.076 |

`submission.csv` is the average of both directions: FID **107.01**, MiFID **0.417**, score (FID + MiFID) / 2 = **53.72**.

Rerunning the same checkpoint through the professor script on Apple MPS gives 54.31: GPU and MPS floating-point results differ slightly, and FID on 300 images amplifies that.

Also in `full_metrics_report.csv`: G/D/cycle/identity losses at epoch 100, grad norm, nan_count=0, params, train time, images/sec, peak memory. `metrics_report.csv` has the same numbers in long format, plus per-epoch losses and every scored checkpoint.

## Checkpoint choice
`score_checkpoints.py` scored the decay-phase checkpoints with the professor method (`outputs/checkpoint_scores_100ep.csv`, Apple MPS): epoch 53 → 58.10, 70 → 56.43, 80 → 55.41, 90 → 54.33, 95 → 54.13, 100 → 54.31. The score improves through the decay and flattens from epoch 90. The submission uses the final epoch (100).

## Human audit
New 30-image sheet for this checkpoint: `outputs/human_audit/audit_30.csv` (same 30 indices as before, two raters). `audit_agreement.py` prints the mean scores and Cohen's kappa once both columns are filled. The earlier audit of the 80-epoch images is kept in `outputs/human_audit/epoch80_archive/`.

## Kaggle
- Team: PairProgramming_Team_5
- Upload file: `submission.csv` (`ID,FID,MiFID` = 1, 107.01382976154575, 0.4169285297393799)
