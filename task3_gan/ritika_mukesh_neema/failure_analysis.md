# Task 3 — Failure Analysis: CycleGAN Photo ↔ Monet (v2)

**Author:** Ritika Mukesh Neema

Submitted checkpoint: `checkpoints/best.pt` (epoch 110, raw weights; TA score 48.0046). The image observations
below are from `outputs/sample_grid.png` (6 fixed rows; columns photo | photo→Monet | Monet | Monet→photo) and the
30 audit images in `outputs/human_audit/`; the numbers are from `full_metrics_report.csv` and `outputs/checkpoint_scores.csv`.

## Case 1: smooth gradients and flat areas get streaks (photo → Monet)

The beach-at-dusk row of the grid is the clearest failure: a smooth pink-to-blue sky becomes an orange-yellow field
covered in vertical streaks, and the soft colour gradient is lost. Flat skies in other rows show a finer version of
the same grain. A Monet painting never has a perfectly smooth area, so the generator fills every flat region with
brush texture, and with nothing in the photo to anchor it the texture turns into repeated streaks.

## Case 2: night and dark scenes (photo → Monet)

The night-mountain row turns a deep blue sky into a muted, textured blue-grey, and the moon becomes a smeared
orange blob. Monet's 300 paintings are almost all daylight scenes, so dark inputs are far from anything the Monet
discriminator has seen and the colours drift toward the daylight palette.

## Case 3: text and watermarks survive as ghosts

The same night photo has a watermark ("DanielMcVey.Com") in the corner, and a faint copy of it is still visible
after translation. The cycle-consistency loss rewards keeping every detail needed to rebuild the photo, so
high-contrast text is carried through instead of being painted over.

## Case 4: Monet → photo keeps painterly texture and narrow coverage

Monet → photo outputs keep the composition well (boats, cliffs, river bank) but still show brush-stroke texture, and
they come out darker and more saturated than real photos, with smeared dark blobs in dark paintings (last row of the
grid). The metrics agree: precision 0.713 but recall only 0.383, i.e. the outputs look photographic but cover a
narrow slice of real photos. This direction has a higher FID (97.3 vs 93.9) and KID (0.0150 vs 0.0046) than
photo → Monet. Its generator only ever sees the same 300 paintings as inputs.

## Case 5: the score is noisy, and the selection uses the evaluation images

Late epochs at a nearly flat learning rate still move by about ±0.5 in score, and Monet → photo FID alone swings by
up to ~3 points between neighbouring epochs (e.g. 96.9 at epoch 96, 99.3 at epoch 98). The submitted checkpoint is
the best of 60 scored candidates on the same 300+300 images the TA script uses, so its score includes some of that
luck. The TA score only uses 300 images per folder: even two sets of real photos score about FID 80 against each
other, which is the floor this metric can reach.

## What worked (for contrast)

Content preservation is strong in both directions: forests, misty lakes and cloudy fields keep their layout and are
convincingly restyled (rows 3, 5, 6 of the grid), and cycle reconstruction L1 is 0.064 / 0.069. Training was stable
throughout (no NaN / Inf in 100,000 steps, discriminator loss 0.26–0.36).

## Process limitations

- Cycle L1 / LPIPS / content cosine are computed on 100 images per direction (`--max_images_for_cycle 100`), so they
  are noisier than FID/KID (300 images).
- The content-cosine metric is computed in pixel space, a cheap proxy that rewards keeping colours as well as content.
- Human audit: the 30 blinded samples are prepared; the two raters' scores and Cohen's kappa are still pending.
