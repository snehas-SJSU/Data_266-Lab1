"""
Task 3.2(6) - Blinded human audit of 30 fixed samples, 2 raters, with
inter-rater agreement (Cohen's kappa + % agreement).

Two steps:
  1. `python human_audit.py prepare`
     Picks 30 fixed (seeded) samples from pred_B2A (the Kaggle-scored
     photo->monet direction), copies them into outputs/human_audit/ with
     BLINDED filenames (sample_01.jpg ... sample_30.jpg -- no info about
     which model/run produced them, so raters aren't biased), and writes
     rater_A_template.csv / rater_B_template.csv for each rater to fill in
     a 1-5 score for style, content-preservation, and artifacts per sample.

  2. Once both raters have filled in their copies (rename to
     rater_A_scores.csv / rater_B_scores.csv), run:
     `python human_audit.py score`
     to compute Cohen's kappa and % exact agreement per criterion, and an
     overall average human audit score.
"""
import argparse
import csv
import json
import os
import random
import shutil

from dataset import list_images

N_SAMPLES = 30
CRITERIA = ["style", "content_preservation", "artifacts"]  # each rated 1 (poor) - 5 (excellent)


def prepare(out_dir, seed=42):
    pred_dir = os.path.join(out_dir, "pred_B2A")
    files = list_images(pred_dir)
    if len(files) < N_SAMPLES:
        raise SystemExit(f"Only {len(files)} images in {pred_dir}, need at least {N_SAMPLES}.")

    rng = random.Random(seed)
    chosen = rng.sample(files, N_SAMPLES)

    audit_dir = os.path.join(out_dir, "human_audit")
    os.makedirs(audit_dir, exist_ok=True)

    mapping = {}
    for i, src in enumerate(sorted(chosen), start=1):
        blinded_name = f"sample_{i:02d}.jpg"
        shutil.copy(src, os.path.join(audit_dir, blinded_name))
        mapping[blinded_name] = os.path.basename(src)

    with open(os.path.join(audit_dir, "_mapping_DO_NOT_SHARE_WITH_RATERS.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    for rater in ["A", "B"]:
        path = os.path.join(audit_dir, f"rater_{rater}_template.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_id"] + CRITERIA + ["notes"])
            for i in range(1, N_SAMPLES + 1):
                w.writerow([f"sample_{i:02d}.jpg", "", "", "", ""])
        print(f"Wrote {path}")

    print(f"\n{N_SAMPLES} blinded images in {audit_dir}/ -- send this folder (minus the mapping "
          f"JSON) plus rater_A_template.csv / rater_B_template.csv to your two raters. "
          f"Score each criterion 1 (poor) - 5 (excellent). When done, save their filled-in "
          f"copies as rater_A_scores.csv and rater_B_scores.csv in {audit_dir}/.")


def _read_scores(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[row["sample_id"]] = {c: int(row[c]) for c in CRITERIA}
    return rows


def cohens_kappa(a, b, labels):
    """Simple unweighted Cohen's kappa for ordinal 1-5 ratings, treated as categorical."""
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for lab in labels:
        pa = sum(1 for x in a if x == lab) / n
        pb = sum(1 for y in b if y == lab) / n
        pe += pa * pb
    if pe == 1.0:
        return 1.0  # degenerate: everyone agrees on everything, kappa undefined -> treat as perfect
    return (po - pe) / (1 - pe)


def score(out_dir):
    audit_dir = os.path.join(out_dir, "human_audit")
    path_a = os.path.join(audit_dir, "rater_A_scores.csv")
    path_b = os.path.join(audit_dir, "rater_B_scores.csv")
    for p in [path_a, path_b]:
        if not os.path.exists(p):
            raise SystemExit(f"Missing {p} -- have both raters filled in their templates yet?")

    scores_a, scores_b = _read_scores(path_a), _read_scores(path_b)
    common = sorted(set(scores_a) & set(scores_b))
    if len(common) < N_SAMPLES:
        print(f"[warn] only {len(common)}/{N_SAMPLES} samples scored by both raters.")

    results = {}
    all_scores = []
    for crit in CRITERIA:
        a = [scores_a[s][crit] for s in common]
        b = [scores_b[s][crit] for s in common]
        pct_agree = sum(1 for x, y in zip(a, b) if x == y) / len(common)
        kappa = cohens_kappa(a, b, labels=[1, 2, 3, 4, 5])
        avg_score = sum(a + b) / (2 * len(common))
        results[crit] = {"pct_agreement": pct_agree, "cohens_kappa": kappa, "mean_score_1_to_5": avg_score}
        all_scores.extend(a + b)

    results["overall_mean_human_audit_score_1_to_5"] = sum(all_scores) / len(all_scores)
    results["n_samples"] = len(common)

    out_path = os.path.join(audit_dir, "human_audit_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "score"])
    ap.add_argument("--out_dir", default="../outputs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mode == "prepare":
        prepare(args.out_dir, args.seed)
    else:
        score(args.out_dir)


if __name__ == "__main__":
    main()
