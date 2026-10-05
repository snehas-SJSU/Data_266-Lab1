# Task 3 — Results: CycleGAN Monet ↔ Photo Style Transfer (v2)

**Author:** Ritika Mukesh Neema

**Submitted checkpoint:** `checkpoints/best.pt` — epoch 110 of the v2 run, raw (non-EMA) weights.
**Official score (TA evaluation script, `evaluate_local.py`):** FID **95.6037**, MiFID **0.4054** → score (FID + MiFID) / 2 = **48.0046** (Kaggle leaderboard shows −48.0046).

Numbers on this page come from these files (none are hand-edited):
`outputs/train_metrics.json`, `outputs/train_log.jsonl` (raw per-step log), `outputs/checkpoint_scores.csv`,
`outputs/loss_curves.png`, `submission.csv`, `full_metrics_report.csv` / `src/full_metrics.json`,
`outputs/console_generate.log`, `outputs/console_evaluate_local.log`, `outputs/console_full_metrics.log`.

## What changed from v1, and why

The v1 run (archived in `outputs/history_v1/`) trained for 80 epochs of only 300 images each. v2 keeps the same
CycleGAN design and fixes what held v1 back:

| | v1 | v2 | Why |
|---|---|---|---|
| Images per epoch | 300 (one pass over the Monet set) | 3,200 (`steps_per_epoch` 800 × batch 4) | v1 saw only 24,000 images per domain in total; v2 sees 400,000 |
| Epochs | 80 (40 + 40 decay) | 125 (50 constant + 75 linear decay) | best checkpoints appear late in the decay phase |
| Batch size | 1 | 4 | more stable updates, better GPU use |
| Generator decoder | transposed convolutions | nearest-neighbour resize + 3×3 conv (`--upsample nearest`) | removes checkerboard artifacts |
| Discriminator augmentation | none | DiffAugment: color, translation, cutout (`augment.py`) | only 300 Monets: stops D_A memorising them |
| Real label | 1.0 | 0.9 (one-sided smoothing) | keeps the discriminators from becoming over-confident |
| Gradient clipping | max norm 10 | off (`grad_clip` 0) | v1's mean gradient norm (57, G + D) was far above the threshold of 10 |
| EMA of generator weights | no | yes, decay 0.999 (`utils.EMA`) | raw and EMA weights are both scored |
| Mixed precision | no | yes (AMP) | faster training, lower memory |
| Checkpoint selection | final epoch | last 30 epochs scored with the TA's notebook, best kept | picks the best epoch on the official metric |
| Saved JPEG quality | 75 (torchvision default) | 95 | fewer compression artifacts in the evaluated images |
| Resume / thermal guard | weights only | full state every epoch (`checkpoints/last.pt`), optional GPU temperature pause | safe on Colab disconnects and laptop runs |

## Architecture

- **Generators** `G_A2B` (Monet → photo) and `G_B2A` (photo → Monet): ResNet-9, 64 base filters, reflection padding,
  instance normalization, two stride-2 downsampling convs, 9 residual blocks, two upsampling stages
  (nearest ×2 + 3×3 conv), tanh output. 11,378,179 parameters each.
- **Discriminators** `D_A` (real vs fake Monet) and `D_B` (real vs fake photo): 70×70 PatchGAN, C64-C128-C256-C512,
  no norm on the first layer. 2,764,737 parameters each.
- **Total:** 28,285,832 parameters. No pretrained weights in any generator or discriminator.

## Training

Settings are in `src/config.json` (command-line flags override it).

| Setting | Value |
|---|---|
| Losses | LSGAN (MSE) adversarial + cycle L1 (λ 10) + identity L1 (weight 5) |
| Optimiser | Adam, lr 2e-4, betas (0.5, 0.999), 50 epochs constant then linear decay to 0 over 75 epochs |
| Batch / steps | batch 4, 800 steps per epoch, 100,000 steps |
| Data | all 300 Monets and all 7,038 photos, 256×256, random horizontal flip; each epoch cycles the Monets ~11 times and draws 3,200 random photos |
| Image pool | 50 fakes per discriminator |
| Seed | 6638 |
| Hardware | Google Colab, NVIDIA A100 80 GB, AMP (fp16) |
| Training time | 16,310 s (4.53 h), 24.5 images/s per domain |
| Peak GPU memory | 17,368 MB (whole run, including the checkpoint scoring passes) |

## Training behaviour and stability

![loss curves](outputs/loss_curves.png)

- Generator, cycle and identity losses fall smoothly over all 125 epochs (G 8.24 → 2.95, cycle 5.00 → 1.34,
  identity 2.28 → 0.55, weighted values). The only visible bump is at epoch 43 (G 4.13, recovered the next epoch).
- The discriminator loss stays in a narrow band, 0.35 early → 0.26 at the end. For reference, two least-squares
  discriminators that only guessed (output 0.5) would sit at about 0.41, and a perfect pair at 0, so the
  discriminators stay useful without overpowering the generators. The adversarial part of the generator loss stays
  near 1.0 throughout — neither side collapses.
- No loss spike when the learning rate starts decaying (epoch 50).
- NaN / Inf events: **0** in 100,000 steps. Gradient norms (G + D, unclipped): mean 32.2, max 469.9.

## Checkpoint selection (TA method)

From epoch 95 on, every epoch was scored with the TA's notebook code (`src/ta_eval.py` loads
`task3_gan/Part3_Evaluation_Script.ipynb`): the first 300 photos → Monet and all 300 Monets → photo, raw and EMA
weights — 60 candidates in `outputs/checkpoint_scores.csv`. Best single checkpoint: **epoch 110, raw weights**
(score 48.0133 at training time; 48.0046 when the full outputs are regenerated and scored).

Because photo → Monet is scored only through `G_B2A` and Monet → photo only through `G_A2B`, the run also saved the
best generator per direction (`best_combo.pt`: photo → Monet from epoch 110 raw, Monet → photo from epoch 121 raw).
Scored the same way it reaches 47.7936. It is not the submitted model; it is listed here because it was measured.

Both selections are made on the images the TA script evaluates, so the chosen score is optimistic by the
epoch-to-epoch noise of this metric (late epochs vary by about ±0.5 in score).

## Metrics (both directions, submitted checkpoint)

**Official (TA script, first 300 images per folder):**

| Direction | FID | MiFID |
|---|---|---|
| Photo → Monet (`pred_B2A`) | 93.892 | 0.3991 |
| Monet → photo (`pred_A2B`) | 97.316 | 0.4118 |
| **Submission (mean)** | **95.6037** | **0.4054** |

**All other metrics** (`src/full_metrics.py`; its own Inception feature extractor, 300 images for FID/KID/precision/recall,
100 images for the cycle metrics):

| Metric | Photo → Monet | Monet → photo |
|---|---|---|
| FID (local extractor) | 92.68 | 98.62 |
| KID (mean ± std over 50 subsets) | 0.0046 ± 0.0011 | 0.0150 ± 0.0023 |
| Precision / recall (k-NN manifold) | 0.477 / 0.637 | 0.713 / 0.383 |
| Cycle-reconstruction L1 (photo→Monet→photo / Monet→photo→Monet) | 0.0693 | 0.0643 |
| LPIPS, input vs cycle reconstruction | 0.142 | 0.194 |
| Content cosine, input vs translation (pixels) | 0.829 | 0.730 |

| Training-side metric | Value |
|---|---|
| Final G / D loss | 2.946 / 0.263 |
| Final cycle / identity loss (weighted) | 1.342 / 0.549 |
| Gradient norm mean / max, NaN count | 32.2 / 469.9, 0 |
| Parameters | 28,285,832 |
| Training time, images/s | 16,310 s, 24.5 |
| Peak memory (GPU / process) | 17,368 MB / 10,199 MB |

## Inference settings

Monet → photo averages 6 flipped/shifted views (`--views_a2b 6`, `src/inference.py`); photo → Monet is a single pass.
Measured during the run on the best generators at the time (TA method): Monet → photo FID 98.43 / 97.19 / 96.92 for
1 / 2 / 6 views; photo → Monet 93.94 with 1 view vs 96.92 with 2 (averaging blurs the brush strokes).

## Cycle-consistency verification

Round trips return close to the input: mean L1 0.064 (Monet → photo → Monet) and 0.069 (photo → Monet → photo)
on a [-1, 1] pixel scale, down from 0.109 / 0.126 in v1. LPIPS between input and reconstruction is 0.19 / 0.14
(v1: 0.405 / 0.337). In `outputs/sample_grid.png` the layout of every scene (horizon, trees, boats, rocks) is kept
in both directions.

## Visual quality

`outputs/sample_grid.png` (columns: photo | photo→Monet | Monet | Monet→photo). See `failure_analysis.md` for the
specific failure cases.

## Human audit

`src/human_audit.py prepare` selected 30 fixed photo→Monet outputs (seed 42) and copied them, blinded, to
`outputs/human_audit/` with `rater_A_template.csv` / `rater_B_template.csv`. **Ratings are pending:** once both raters
fill them in, `python human_audit.py score` reports the mean scores and Cohen's kappa / % agreement.

## Kaggle

- Upload file: `submission.csv` (`ID,FID,MiFID` = `1, 95.60372227321069, 0.4054316282272339`)
- Leaderboard score: −48.0046 (−(FID + MiFID) / 2)
- Rank: to be recorded after upload
- The images behind it are the direct output of this CycleGAN (`src/generate.py` from `checkpoints/best.pt`).

## How to reproduce

```bash
cd task3_gan/ritika_mukesh_neema/src
python train.py                                  # settings from config.json; resumes from ../checkpoints/last.pt
python generate.py --ckpt ../checkpoints/best.pt # outputs/pred_A2B (300), outputs/pred_B2A (7,038)
cd .. && python evaluate_local.py                # TA script -> submission.csv
cd src && python full_metrics.py --ckpt ../checkpoints/best.pt   # -> full_metrics_report.csv, src/full_metrics.json
python sample_grid.py && python human_audit.py prepare
```
One command: `python run_all.py` (smoke test: `python run_all.py --smoke_test`). On Colab the run used
`python train.py --ckpt_dir <Drive>/checkpoints --out_dir <Drive>/outputs`.

## Hardware disclosure

Training: Google Colab, NVIDIA A100-SXM4 80 GB. Generation, TA scoring and metrics for the submitted checkpoint:
NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), Windows 11; the scores match the training-time scores within 0.01.

## v1 (previous run, for reference)

80 epochs × 300 images, batch 1, transposed-conv decoder, no augmentation / EMA, RTX 5090, 27.7 min. Local FID
123.70 (photo → Monet) / 120.72 (Monet → photo); FID 104.30 / MiFID 0.413 with `kaggle_score.py` against
`real_stats.npz` (a different scorer, not comparable with the TA script). All v1 files are in `outputs/history_v1/`.
