# Checkpoints (Ritika Mukesh Neema, Part 3 v2)

The `.pt` files are not in git (`.gitignore`: `task3_gan/**/checkpoints/*.pt`; each is 45–430 MB).
They are on Google Drive: https://drive.google.com/drive/folders/1UCvPH_Z5sNjOiCQ-IjDyEJippmuAtNNv?usp=sharing

| File | What it is | Size |
|---|---|---|
| `best.pt` | **Submitted model**: epoch 110, raw weights (`G_A2B`, `G_B2A`, run config). TA score 48.0046 | 91 MB |
| `best_combo.pt` | Best generator per direction: photo→Monet from epoch 110 raw, Monet→photo from epoch 121 raw. TA score 47.7936 (not submitted) | 91 MB |
| `best_G_B2A.pt`, `best_G_A2B.pt` | The two generators in `best_combo.pt`, as separate state dicts | 46 MB each |
| `ckpt_final.pt` | Epoch 124 (end of training): raw + EMA generators and both discriminators | 204 MB |
| `last.pt` | Full training state for resuming (models, EMA, optimizers, schedulers, AMP scalers, history, RNG) | 432 MB |

Place them in this folder, then from `src/`:

```bash
python generate.py --ckpt ../checkpoints/best.pt
cd .. && python evaluate_local.py
```

All checkpoints store their architecture in `config` (64 filters, 9 residual blocks, nearest upsampling); `src/inference.py`
rebuilds the generators from it. 28,285,832 parameters in total (G_A2B + G_B2A + D_A + D_B).
The v1 checkpoint (`ckpt_final.pt`, 80 epochs) is at https://drive.google.com/file/d/1Wlu22EKbp4QATijRbNBbDOFFGHLHUiNj/view
