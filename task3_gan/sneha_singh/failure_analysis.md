# Part 3 — Failure / error analysis (Sneha Singh)

I looked at the full-run outputs from Colab (Tesla T4, 80 epochs): loss curves,
the sample grid, a bunch of `pred_A2B` images, and the numbers from
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
| A2B | 0.117 | 0.436 | 0.576 | 95.13 | 0.411 |
| B2A | 0.102 | 0.329 | 0.632 | 96.36 | 0.433 |

Cycle L1 around 0.1 feels fine after training (cycle loss went from ~7 → ~2.2).
A2B content cosine ~0.58 matches what I see: scene is there, not pixel-perfect.
Full B2A (300) looks healthier than the old smoke numbers — LPIPS 0.33 and
content cos 0.63, close to A2B.

## Training / loss curves

Looking at `outputs/loss_curves/losses.png`:

- G, cycle, and identity all go down smoothly. Training didn’t stall.
- D stays really low the whole time (~0.2). Discriminator is winning a lot —
  that might be why textures get noisy instead of clean brushstrokes.
- Grad norms bounce around but don’t explode. `nan_count = 0`.
- We only did 80 epochs (lab budget), not the paper’s 200, so leftover artifacts
  aren’t shocking.

## MiFID / memorization

Local A2B: FID **95.13**, MiFID **0.411**.
When I rated, images looked like stylized versions of that photo, not pasted
Monet paintings. The main issue is shared grainy texture across many outputs.
Ritika and I will confirm that on the joint sheet.

Scores came from our local `evaluate_local.py`. We still need the course
`real_stats.npz` / eval script before treating them as final leaderboard numbers.

## Next steps if we retrain

1. More epochs or slightly higher identity loss.
2. Change upsampling to reduce checkerboard.
3. Finish joint 30 ratings and record agreement here.
