# Part 3 checkpoints (Sneha Singh)

`best.pt` is 113 MB, over GitHub’s 100 MB file limit, so the weights are not in git.

- **Local path:** `task3_gan/sneha_singh/checkpoints/best.pt` (on this machine / after download)
- **Link:** https://drive.google.com/drive/folders/12AgM95RbZAQUyouH7sukM9nu_rVTD3C9?usp=share_link
- **Smoke:** `best_smoke.pt` was a local MPS smoke run only; not kept on git

`best.pt` is epoch 100 of the 100-epoch run (50 constant + 50 decay, Colab NVIDIA L4, AMP, ~4.0 h). `submission.csv` is the professor average for this checkpoint: FID 107.01, MiFID 0.417 (score 53.72).
