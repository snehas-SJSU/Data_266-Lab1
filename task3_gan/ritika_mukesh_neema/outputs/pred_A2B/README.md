# pred_A2B — Monet → photo

300 generated photos (one per Monet painting, same filenames), from `checkpoints/best.pt` (epoch 110); each output is the
average of 6 flipped/shifted views. TA evaluation (`evaluate_local.py`): FID 97.316, MiFID 0.4118.

All 300 images are committed (force-added past `.gitignore`), so `evaluate_local.py` can be re-run straight from the repo.
Rebuild from `src/`: `python generate.py --ckpt ../checkpoints/best.pt`
