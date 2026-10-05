# Part 3 — Failure / error analysis (Sneha Singh)

Submitted checkpoint: epoch 91 of the 100-epoch run (50 + 50 decay, 64 base filters, DiffAugment, EMA, identity weight 5, AMP).
The image observations below are from my audit of an earlier run of the same CycleGAN; the 30-image audit of the submitted
images is in `outputs/human_audit/audit_30.csv`. Numbers are for the epoch-91 checkpoint.

## What usually goes wrong in the images

From `outputs/samples/grid.png` and the audit set:

1. **Content drift** — big shapes (fields, horizon, trees) usually stay. Small
   details get soft. Monet→photo in the grid looks blurry / soft-focus.
2. **Color wash** — some skies go flat or muddy (storm clouds turn into a pale
   band). Sunsets can brown out at the edges.
3. **Checkerboard / grain** — this is the one I see most. Flat areas (sky, water)
   often have a fine grid or noisy dots. Classic GAN upsample junk.
4. **Same brush texture over and over** — not full mode collapse (scenes still
   look different), but a lot of outputs share the same grainy “dab” look instead
   of varied Monet strokes.
5. **Still looks like a photo** — less common after training. More often it’s
   *too* noisy than not stylized enough.
6. **Watermarks / text** — a few preds show smeared white blobs where the photo
   probably had text on it.

## 30-sample audit

- Rater: me, on images from an earlier checkpoint of this CycleGAN (80-epoch run, same 30 indices)
- Sheet and images: `outputs/human_audit/audit_30.csv`, `a2b_*.jpg`
- Scale: style 0–2, content 0–2, artifacts 0–2 (higher = worse), mem Y/N

### My pass (quick summary)

| What I saw a lot | How often |
|---|---|
| Grain / grid in sky or water | Most images |
| Layout of the photo still readable | Most images |
| Strong Monet feel without heavy junk | Some (e.g. blossom tree, lighthouse) |
| Basically broken / band noise | A few (e.g. 01455, 04853) |
| Looks like a copied training Monet | None that I flagged (all N) |

Means from `audit_30.csv` (30 images, 0–2): style 1.40, content 1.67, artifacts 1.43.
None of the 30 is marked memorised. The second rater’s columns are blank, so there is no kappa.
Content is the strongest part; artifacts are the main complaint.

## Cycle check

From `full_metrics_report.csv`:

| | FID | MiFID |
|---|---|---|
| A2B (Monet→photo) | 94.83 | 0.412 |
| B2A (photo→Monet) | 95.78 | 0.406 |

Cycle loss went from ~5.0 at epoch 1 to ~1.42 at epoch 91.

## Training / loss curves

Looking at `outputs/loss_curves/losses.png`:

- G, cycle, and identity all go down smoothly. Training didn’t stall.
- D stays around 0.26–0.31 for the first 40 epochs, then falls to about 0.22 by epoch 91. Discriminator still wins
  late in training — that might be why textures get noisy instead of clean brushstrokes.
- Epoch 50 has one spike (D 0.65) right before the learning-rate decay starts; it recovers by epoch 51.
- Grad norms bounce around but don’t explode. `nan_count = 0`.
- We did 100 epochs (lab budget), not the paper’s 200, so leftover artifacts
  aren’t shocking.

## MiFID / memorization

Professor script (GPU), both directions: A2B FID **94.83**, MiFID **0.412**. B2A FID **95.78**, MiFID **0.406**.
`submission.csv` is the average: FID **95.30**, MiFID **0.409** (score 47.857).
When I rated, images looked like stylized versions of that photo, not pasted
Monet paintings. The main issue is shared grainy texture across many outputs.

Scores came from `evaluate_local.py`, which follows the professor notebook (Inception-v3, first 300 sorted images, mean cosine distance).

## Next steps if we retrain

Epochs 85–100 score between 47.8 and 49.4 (`outputs/checkpoint_scores_epochs76_100.csv`), so the model has plateaued.
Averaging 6 flipped/shifted views lowers Monet→photo FID from 96.5 to 94.8; for photo→Monet it does not help, so that direction stays single pass.
