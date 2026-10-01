# Task 3 — Failure Analysis: CycleGAN Photo ↔ Monet

**Author:** Ritika Mukesh Neema

Two real, metric-grounded failure modes observed in this run's local evaluation (`outputs/full_metrics_report.json`), plus one process limitation.

## Case 1: Low precision / high recall asymmetry in the photo→Monet direction (B2A)

The photo→Monet direction — the one Kaggle actually scores — has precision 0.327 but recall 0.570 (improved precision/recall, Kynkaanniemi et al. 2019, k=3, 300-image sample). Recall this high relative to precision means many of the *real* Monet images' neighborhoods are being reached by *some* generated sample (good mode coverage), but a large fraction of individual generated images fall outside any real image's local neighborhood in Inception feature space (low fidelity per-sample). Concretely: the generator produces a reasonably diverse spread of outputs, but a sizeable share of those outputs are not convincing Monet-style images on their own.

This is consistent with a model that hasn't trained long enough to sharpen its mapping — cycle-consistency and identity losses can keep outputs roughly in the right global distribution (driving recall up) well before the discriminator has forced individual samples to be locally realistic (which would raise precision). The other direction, monet→photo (A2B), shows the flipped pattern (precision 0.613, recall 0.207): fewer modes covered, but the outputs that are produced are closer to real photos. This is the expected asymmetry for a 300-image domain (Monet) vs. a 7,038-image domain (photo) — there is far more real-photo structure for A2B's discriminator to enforce against, while B2A's target domain (Monet) is small and its discriminator saturates faster without stabilizing fine style texture across the output distribution.

## Case 2: Training stopped well short of convergence given KID's variance

KID (bootstrapped over 50 subsets) for B2A is 0.0267 ± 0.0029 — a non-trivial standard deviation relative to the mean, which (together with final generator loss still at 4.84, not clearly plateaued) indicates the model was still improving when training stopped at 80 total epochs (40 + 40 decay). The original CycleGAN paper trains 200 epochs; this run used 40% of that budget.

## Process limitation (not a model-quality failure)

Cycle-reconstruction L1/LPIPS/content-cosine are computed on only 100 images per direction (`--max_images_for_cycle 100`, `evaluate_local.py`), not the full dataset — a deliberate speed/cost tradeoff for local evaluation. This means the cycle-consistency numbers in `metrics_report.csv` carry more sampling noise than the FID/KID numbers (which use up to 300 images).

## Pending: qualitative/visual failure inspection

A human visual audit of 30 blinded photo→Monet samples (2 raters, Cohen's kappa) is specified in the assignment and tracked in `src/human_audit.py` / `outputs/human_audit/` — this has not been run yet. Specific per-image failure examples (e.g., color bleed, texture artifacts, structural distortion) will be added here once that audit is complete, rather than asserted without having actually looked at the generated images.
