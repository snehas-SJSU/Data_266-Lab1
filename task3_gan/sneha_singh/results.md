# Part 3 — CycleGAN photo ↔ Monet (Sneha Singh)

Dataset: unpaired Monet paintings (300) and photos (7,038), Kaggle. Own CycleGAN only — no pretrained image models on outputs.

## Architecture
- Generators: ResNet, 9 residual blocks at 256×256, 64 base filters, reflection pad, instance norm, tanh output; upsampling is nearest-neighbour ×2 then a stride-1 conv
- Discriminators: 70×70 PatchGAN (LSGAN / MSE), real labels smoothed to 0.9
- DiffAugment (colour, translation, cutout) on every image the discriminators see, real and fake
- Losses: adversarial + cycle-consistency (λ = 10) + identity (weight 5)
- Image pool of 50; Adam (β1 = 0.5), learning rate 2e-4 for G and D
- 28.3M parameters (all four networks)

## Training
- 100 epochs: 50 at constant learning rate, 50 with linear decay to 0
- 800 steps per epoch, batch 4 (3,200 photos resampled each epoch, all 300 Monet cycled)
- Epochs 1–84: Colab NVIDIA A100. Epochs 85–100: resumed on an NVIDIA RTX 4060 Laptop GPU. Mixed precision, 19,836 s (~5.5 h), peak GPU memory 6,855 MB
- Notebook: `src/part3_cyclegan.ipynb`; config: `src/config.json`
- Raw log: `reproducibility/raw_logs/sneha_singh/task3_gan/train_full_ngf64_sneha.log`

## Checkpoint and inference
- Epochs 76–100 were scored with the professor evaluation (`outputs/checkpoint_scores_epochs76_100.csv`). Epoch 91 scored best and is `checkpoints/best.pt`.
- Monet→photo (`pred_A2B`): output averaged over 6 views (original, horizontal flip, four 4-pixel shifts). Photo→Monet (`pred_B2A`): single pass.
- `make_submission.py` regenerates both folders and `outputs/samples/grid.png` from `best.pt`; `evaluate_local.py` then writes `submission.csv` and `full_metrics_report.csv`.

## Metrics
FID and MiFID: unchanged professor evaluation script (Inception-v3, first 300 sorted images, MiFID = mean cosine distance), run on GPU.

| Direction | FID | KID | MiFID | Precision | Recall | LPIPS | content cos | cycle L1 |
|---|---|---|---|---|---|---|---|---|
| A2B (Monet→photo) | 94.83 | – | 0.412 | – | – | – | – | – |
| B2A (photo→Monet) | 95.78 | – | 0.406 | – | – | – | – | – |

`submission.csv`: FID **95.30**, MiFID **0.409**, score (FID + MiFID) / 2 = **47.857**.

Losses at epoch 91: G 3.191, D 0.224, cycle 1.420, identity 0.589, NaN count 0. All values are in `full_metrics_report.csv`; `metrics_report.csv` has the same in long format plus per-epoch losses and every scored checkpoint.

## Human audit
`outputs/human_audit/audit_30.csv`: 30 fixed photo→Monet images (`a2b_*.jpg` in that folder), rated on an earlier checkpoint of this CycleGAN (80-epoch run, same 30 indices); not repeated for the submitted epoch-91 checkpoint. My means (1–5, higher = better, artifacts 5 = clean): style 3.80, content 4.33, artifacts 2.13, overall 3.42. These are the original 0–2 marks placed on 1–5. 0 of 30 marked as memorised. Ritika rated the same 30 images: style 3.87, content 4.07, artifacts 2.00, overall 3.31. Cohen’s kappa: style 0.560, content 0.591, artifacts 0.600 (exact agreement 76.7% / 80.0% / 80.0%); both raters marked none as memorised.

## Kaggle
- Team: PairProgramming_Team_05
- Upload file: `submission.csv` (`ID,FID,MiFID` = 1, 95.30491589533982, 0.4091329425573349)
- Kaggle score of this submission: 47.857 (earlier upload 50.1253); team leaderboard rank 9
