# Part 3 — CycleGAN (Sneha Singh)

Own CycleGAN. Own Kaggle submission (required individually).

## Folder map (PDF)

```
task3_gan/
  data/
    monet_jpg/          # shared raw Monet (Drive, not git)
    photo_jpg/          # shared raw photos (Drive, not git)
  sneha_singh/
    src/                # part3_cyclegan.ipynb + config.json
    data_processed/     # my preprocessing only
    checkpoints/        # G_AB, G_BA, D_A, D_B weights
    outputs/
      pred_A2B/         # photo → Monet
      pred_B2A/         # Monet → photo
      loss_curves/
      samples/
      human_audit/      # 30 fixed samples for 2-rater audit
    evaluate_local.py
    submission.csv      # Kaggle submit artifact (or images.zip note)
    full_metrics_report.csv
    failure_analysis.md
    results.md
```

## Architecture
- Generators (A→B, B→A): _fill when coding_
- Discriminators: _fill when coding_
- Losses: adversarial + cycle-consistency (+ identity if used)

## Hyperparameters
- See `src/config.json`
- Epochs / batch / LR / λ_cycle: _fill after train_

## Kaggle
- Competition: class signup (Monet / “Painter Myself” style)
- Team name:
- Submission must be **this model’s inference only**
- Public / private score:
- Leaderboard rank:

## Metrics (both directions)
See `full_metrics_report.csv`: FID, KID, precision/recall, cycle L1, LPIPS, content cosine, losses, human audit + agreement, params, time, peak memory, Kaggle scores.
