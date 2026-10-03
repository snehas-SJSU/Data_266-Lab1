# Part 3 checkpoints (Sneha Singh)

`best.pt` (43 MB) is in git: epoch 93 of the 100-epoch run (50 constant + 50 decay, Colab NVIDIA L4, AMP, ~5.4 h).
It holds both generators and both discriminators. Generator: 9 residual blocks, 32 base filters (`ngf: 32` in `src/config.json`).

- Regenerate the submitted outputs: `python task3_gan/sneha_singh/make_submission.py`, then `python task3_gan/sneha_singh/evaluate_local.py`
- `submission.csv` for this checkpoint: FID 99.84, MiFID 0.412 (score 50.13)
