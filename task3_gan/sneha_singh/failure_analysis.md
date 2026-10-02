# Part 3 — Failure / error analysis (Sneha Singh)

I looked at this 80-epoch retrain (nearest upsample, label smoothing 0.9, CUDA + AMP): loss curves,
the sample grid, the photo→Monet images in `pred_B2A`, and the numbers from
`evaluate_local.py`. Ritika still needs to fill her half of the 30-sample sheet
before we can write agreement (κ).

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
5. **Still looks like a photo** — less common after 80 epochs. More often it’s
   *too* noisy than not stylized enough.
6. **Watermarks / text** — a few preds show smeared white blobs where the photo
   probably had text on it.

## 30-sample audit

- Raters: me (done solo), Ritika (pending)
- Sheet: `outputs/human_audit/audit_30.csv` (same 30 files for both of us)
- Scale: style 0–2, content 0–2, artifacts 0–2 (higher = worse), mem Y/N
- Inter-rater agreement: _fill after Ritika rates_

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
| A2B (Monet→photo) | 0.103 | 0.365 | 0.586 | 109.33 | 0.425 |
| B2A (photo→Monet) | 0.112 | 0.452 | 0.532 | 105.06 | 0.405 |

Cycle L1 around 0.11 feels fine after training (cycle loss went from ~5.9 → ~2.17).
B2A content cosine ~0.53 matches what I see on photo→Monet: scene is there, not pixel-perfect.
A2B (300 Monet→photo images) is in the same range — LPIPS 0.37 and content cos 0.59.

## Training / loss curves

Looking at `outputs/loss_curves/losses.png`:

- G, cycle, and identity all go down smoothly. Training didn’t stall.
- D stays really low and ends near 0.17. Discriminator is winning a lot —
  that might be why textures get noisy instead of clean brushstrokes.
- Grad norms bounce around but don’t explode. `nan_count = 0`.
- We only did 80 epochs (lab budget), not the paper’s 200, so leftover artifacts
  aren’t shocking.

## MiFID / memorization

Professor script, both directions: A2B FID **109.33**, MiFID **0.425**. B2A FID **105.06**, MiFID **0.405**.
`submission.csv` is the average: FID **107.20**, MiFID **0.415**.
When I rated, images looked like stylized versions of that photo, not pasted
Monet paintings. The main issue is shared grainy texture across many outputs.
Ritika and I will confirm that on the joint sheet.

Scores came from `evaluate_local.py`, which follows the professor notebook (Inception-v3, first 300 sorted images, mean cosine distance).

## Next steps if we retrain

1. Later full-photo retrains scored worse than this checkpoint, so `submission.csv` stays on this 80-epoch run.
2. Finish joint 30 ratings and record agreement here.
