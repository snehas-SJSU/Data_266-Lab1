# Part 3 checkpoints (Sneha Singh)

`best.pt` (87 MB) is in git: epoch 91, raw weights, of the 100-epoch run (50 constant + 50 decay, epochs 1–84 Colab NVIDIA A100, 85–100 resumed on an NVIDIA RTX 4060 Laptop GPU, AMP, ~5.5 h).
It holds both generators. Generator: 9 residual blocks, 64 base filters (`ngf: 64` in `src/config.json`).

- Regenerate the submitted outputs: `python task3_gan/sneha_singh/make_submission.py`, then `python task3_gan/sneha_singh/evaluate_local.py`
- `submission.csv` for this checkpoint: FID 95.30, MiFID 0.409 (score 47.857)
