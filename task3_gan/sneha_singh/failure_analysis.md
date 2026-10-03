# Part 3 — Failure / error analysis (Sneha Singh)

Submitted checkpoint: epoch 93 of the 100-epoch run (50 + 50 decay, 32 base filters, DiffAugment, identity weight 5, Colab L4 + AMP).
The image observations below are from my audit of an earlier run of the same CycleGAN; the 30-image audit of the submitted
images is in `outputs/human_audit/audit_30.csv`. Numbers are for the epoch-93 checkpoint.

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

- Rater: me (earlier run, same 30 indices; sheet in git history)
- Submitted-model sheet (two raters): `outputs/human_audit/audit_30.csv`
- Scale: style 0–2, content 0–2, artifacts 0–2 (higher = worse), mem Y/N

### My pass (quick summary)

| What I saw a lot | How often |
|---|---|
| Grain / grid in sky or water | Most images |
| Layout of the photo still readable | Most images |
| Strong Monet feel without heavy junk | Some (e.g. blossom tree, lighthouse) |
| Basically broken / band noise | A few (e.g. 01455, 04853) |
| Looks like a copied training Monet | None that I flagged (all N) |

Average-ish from my scores: style ~1.4, content ~1.7, artifacts ~1.4.
So content is the strongest part; artifacts are the main complaint.

## Cycle check

From `full_metrics_report.csv`:

| | cycle L1 | LPIPS vs source | content cosine vs source | FID | MiFID |
|---|---|---|---|---|---|
| A2B (Monet→photo) | 0.083 | 0.360 | 0.636 | 101.39 | 0.418 |
| B2A (photo→Monet) | 0.083 | 0.396 | 0.595 | 98.29 | 0.406 |

Cycle L1 is about 0.083 at epoch 93 (cycle loss went from ~4.7 → ~1.64).
B2A content cosine is 0.60 and A2B 0.64 (LPIPS 0.40 and 0.36).

## Training / loss curves

Looking at `outputs/loss_curves/losses.png`:

- G, cycle, and identity all go down smoothly. Training didn’t stall.
- D stays around 0.25–0.31 for the first 40 epochs, then falls to about 0.11 by epoch 93. Discriminator is winning a lot
  late in training — that might be why textures get noisy instead of clean brushstrokes.
- Epoch 52 has one spike (D 1.19) where the learning-rate decay starts; it recovers by epoch 54.
- Grad norms bounce around but don’t explode. `nan_count = 0`.
- We did 100 epochs (lab budget), not the paper’s 200, so leftover artifacts
  aren’t shocking.

## MiFID / memorization

Professor script (Colab L4), both directions: A2B FID **101.39**, MiFID **0.418**. B2A FID **98.29**, MiFID **0.406**.
`submission.csv` is the average: FID **99.84**, MiFID **0.412** (score 50.13).
When I rated, images looked like stylized versions of that photo, not pasted
Monet paintings. The main issue is shared grainy texture across many outputs.

Scores came from `evaluate_local.py`, which follows the professor notebook (Inception-v3, first 300 sorted images, mean cosine distance).

## Next steps if we retrain

Epochs 86–100 score between 51.1 and 52.1 with single-pass inference (`outputs/checkpoint_scores_epochs86_100.csv`), so the model has plateaued.
Averaging 6 flipped/shifted views lowers Monet→photo FID from 105.2 to 101.4; the same averaging raises photo→Monet FID, so that direction stays single pass.
Fine-tuning from epoch 93 with a generator EMA reached 50.77 (`outputs/checkpoint_scores_finetune_ema.csv`), so the submission stays on epoch 93.
