# pred_B2A — photo → Monet (Kaggle-scored direction)

7,038 generated Monet-style images (one per photo, same filenames), from `checkpoints/best.pt` (epoch 110), single pass.
TA evaluation (`evaluate_local.py`, first 300 sorted): FID 93.892, MiFID 0.3991.

Committed: the first 300 images in sorted filename order — exactly the ones the TA script scores (force-added past
`.gitignore`). The full set of 7,038 (251 MB) is too large for git; rebuild it from `src/`:
`python generate.py --ckpt ../checkpoints/best.pt`
