# pred_A2B — Monet → photo

300 generated photos (one per Monet painting, same filenames), from `checkpoints/best.pt` (epoch 110); each output is the
average of 6 flipped/shifted views. TA evaluation (`evaluate_local.py`): FID 97.316, MiFID 0.4118.

Not committed (`.gitignore`). Rebuild from `src/`: `python generate.py --ckpt ../checkpoints/best.pt`
