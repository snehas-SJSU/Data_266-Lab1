# pred_B2A — photo → Monet (Kaggle-scored direction)

7,038 generated Monet-style images (one per photo, same filenames), from `checkpoints/best.pt` (epoch 110), single pass.
TA evaluation (`evaluate_local.py`, first 300 sorted): FID 93.892, MiFID 0.3991.

Not committed (`.gitignore`). Rebuild from `src/`: `python generate.py --ckpt ../checkpoints/best.pt`
