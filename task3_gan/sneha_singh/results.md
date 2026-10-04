# Part 3 — CycleGAN photo ↔ Monet (Sneha Singh)

Dataset: unpaired Monet paintings (300) and photos (7,038), Kaggle. Own CycleGAN only — no pretrained image models on outputs.

## Architecture
- Generators: ResNet, 9 residual blocks at 256×256, 32 base filters, reflection pad, instance norm, tanh output; upsampling is nearest-neighbour ×2 then a stride-1 conv
- Discriminators: 70×70 PatchGAN (LSGAN / MSE), real labels smoothed to 0.9
- DiffAugment (colour, translation, cutout) on every image the discriminators see, real and fake
- Losses: adversarial + cycle-consistency (λ = 10) + identity (weight 5)
- Image pool of 50; Adam (β1 = 0.5), learning rate 2e-4 for G and D
- 11.2M parameters (all four networks)

## Training
- 100 epochs: 50 at constant learning rate, 50 with linear decay to 0
- 800 steps per epoch, batch 4 (3,200 photos resampled each epoch, all 300 Monet cycled)
- Colab NVIDIA L4, mixed precision, 19,483 s (~5.4 h), peak GPU memory 3,361 MB
- Notebook with outputs: `src/part3_cyclegan.ipynb`; config: `src/config.json`
- Raw log: `reproducibility/raw_logs/sneha_singh/task3_gan/train_full_light.log`

## Checkpoint and inference
- The last 15 epochs were scored with the professor evaluation (`outputs/checkpoint_scores_epochs86_100.csv`). Epoch 93 scored best and is `checkpoints/best.pt`.
- Monet→photo (`pred_A2B`): output averaged over 6 views (original, horizontal flip, four 4-pixel shifts). Photo→Monet (`pred_B2A`): single pass.
- `make_submission.py` regenerates both folders and `outputs/samples/grid.png` from `best.pt`; `evaluate_local.py` then writes `submission.csv` and `full_metrics_report.csv`.

## Metrics
FID and MiFID: unchanged professor evaluation script (Inception-v3, first 300 sorted images, MiFID = mean cosine distance), run on Colab L4.

| Direction | FID | KID | MiFID | Precision | Recall | LPIPS | content cos | cycle L1 |
|---|---|---|---|---|---|---|---|---|
| A2B (Monet→photo) | 101.39 | 0.019 | 0.418 | 0.747 | 0.078 | 0.360 | 0.636 | 0.083 |
| B2A (photo→Monet) | 98.29 | 0.012 | 0.406 | 0.679 | 0.120 | 0.396 | 0.595 | 0.083 |

`submission.csv`: FID **99.84**, MiFID **0.412**, score (FID + MiFID) / 2 = **50.13**.

Losses at epoch 93: G 3.96, D 0.112, cycle 1.645, identity 0.739, NaN count 0. All values are in `full_metrics_report.csv`; `metrics_report.csv` has the same in long format plus per-epoch losses and every scored checkpoint.

## Human audit
`outputs/human_audit/audit_30.csv`: 30 fixed photo→Monet images (`a2b_*.jpg` in that folder), rated on an earlier checkpoint of this CycleGAN (80-epoch run, same 30 indices); not repeated for the final epoch-93 checkpoint. Sneha's means (0–2): style 1.40, content 1.67, artifacts 1.43 (higher = worse), 0 of 30 marked as memorised. `audit_agreement.py` prints both raters' means and Cohen's kappa once Ritika's column is filled.

## Kaggle
- Team: PairProgramming_Team_05
- Upload file: `submission.csv` (`ID,FID,MiFID` = 1, 99.8386168442841, 0.41217851638793945)
- Kaggle score of this submission: 50.1253 
