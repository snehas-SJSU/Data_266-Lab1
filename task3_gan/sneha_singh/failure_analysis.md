# Part 3 — Failure / error analysis (Sneha Singh)

Submitted checkpoint: the 100-epoch run (50 + 50 decay, nearest upsample, label smoothing 0.9, identity weight 50, Colab L4 + AMP).
The image observations below are from my audit of the earlier 80-epoch run (same architecture and upsampling);
the 30-image audit of the epoch-100 images is in `outputs/human_audit/audit_30.csv`. Numbers are for the epoch-100 checkpoint.

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
5. **Still looks like a photo** — less common after 80 epochs (earlier run). More often it’s
   *too* noisy than not stylized enough.
6. **Watermarks / text** — a few preds show smeared white blobs where the photo
   probably had text on it.

## 30-sample audit

- Rater: me (earlier 80-epoch images)
- Sheet: `outputs/human_audit/epoch80_archive/audit_30_epoch80.csv`
- Epoch-100 sheet (two raters): `outputs/human_audit/audit_30.csv`
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
| A2B (Monet→photo) | 0.089 | 0.274 | 0.688 | 108.12 | 0.422 |
| B2A (photo→Monet) | 0.076 | 0.279 | 0.717 | 105.91 | 0.412 |

Cycle L1 is 0.08–0.09 at epoch 100 (cycle loss went from ~5.5 → ~1.68).
B2A content cosine is 0.72 and A2B 0.69 (LPIPS 0.28 and 0.27).

## Training / loss curves

Looking at `outputs/loss_curves/losses.png`:

- G, cycle, and identity all go down smoothly. Training didn’t stall.
- D stays really low and ends near 0.094 at epoch 100 (0.17 in the 80-epoch run). Discriminator is winning a lot —
  that might be why textures get noisy instead of clean brushstrokes.
- Grad norms bounce around but don’t explode. `nan_count = 0`.
- We did 100 epochs (lab budget), not the paper’s 200, so leftover artifacts
  aren’t shocking.

## MiFID / memorization

Professor script (Colab L4), both directions: A2B FID **108.12**, MiFID **0.422**. B2A FID **105.91**, MiFID **0.412**.
`submission.csv` is the average: FID **107.01**, MiFID **0.417** (score 53.72).
When I rated, images looked like stylized versions of that photo, not pasted
Monet paintings. The main issue is shared grainy texture across many outputs.

Scores came from `evaluate_local.py`, which follows the professor notebook (Inception-v3, first 300 sorted images, mean cosine distance).

## Next steps if we retrain

`outputs/checkpoint_scores_100ep.csv` scores epochs 53–100 of this run: the score improves through the LR decay and flattens from epoch 90 (54.1–54.3 on Apple MPS). Weight averaging (epochs 90/95/100) and flip test-time augmentation moved it by less than 0.3, so `submission.csv` uses the plain final epoch.
