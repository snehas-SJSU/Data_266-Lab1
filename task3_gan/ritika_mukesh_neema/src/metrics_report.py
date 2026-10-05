"""
Task 3 required metrics in one file: ../metrics_report.csv (long format: metric, direction, value, source).

Collects, without recomputing anything:
  - FID / MiFID both directions + submission score  <- ../outputs/ta_scores.json   (TA script, ../evaluate_local.py)
  - KID, precision/recall, cycle L1, LPIPS, content cosine  <- full_metrics.json       (full_metrics.py)
  - losses, gradient norms / NaN, parameters, time, images/s, memory  <- ../outputs/train_metrics.json
  - best scored checkpoint  <- ../outputs/checkpoint_scores.csv
  - human audit means + Cohen's kappa / % agreement  <- ../outputs/human_audit/human_audit_results.json (if scored)
  - Kaggle leaderboard rank: pass --kaggle_rank once known

Usage (after evaluate_local.py and full_metrics.py):
    python metrics_report.py [--kaggle_rank N]
"""
import argparse
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MEMBER = os.path.dirname(HERE)
OUT = os.path.join(MEMBER, "outputs")


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle_rank", default="", help="leaderboard rank, once the submission is uploaded")
    args = ap.parse_args()

    ta = load(os.path.join(OUT, "ta_scores.json"))
    fm = load(os.path.join(HERE, "full_metrics.json"))
    tm = load(os.path.join(OUT, "train_metrics.json"))
    scores = list(csv.DictReader(open(os.path.join(OUT, "checkpoint_scores.csv"))))
    best = min(scores, key=lambda r: float(r["score"]))
    cyc_b, cyc_a = fm["cycle_B_photo_monet_photo"], fm["cycle_A_monet_photo_monet"]
    TA, FM, TR = "evaluate_local.py (TA script)", "src/full_metrics.py", "outputs/train_metrics.json"

    rows = [
        ("FID", "photo->Monet (B2A)", ta["fid_B2A"], TA),
        ("FID", "Monet->photo (A2B)", ta["fid_A2B"], TA),
        ("MiFID", "photo->Monet (B2A)", ta["mifid_B2A"], TA),
        ("MiFID", "Monet->photo (A2B)", ta["mifid_A2B"], TA),
        ("submission FID (mean of both)", "both", ta["fid"], TA + " -> submission.csv"),
        ("submission MiFID (mean of both)", "both", ta["mifid"], TA + " -> submission.csv"),
        ("score (FID + MiFID) / 2", "both", ta["score"], "leaderboard shows the negative"),
        ("Kaggle leaderboard score", "both", -ta["score"], "-(FID + MiFID) / 2 of submission.csv"),
        ("Kaggle leaderboard rank", "both", args.kaggle_rank or "pending", "fill with --kaggle_rank"),
        ("KID mean", "photo->Monet (B2A)", fm["kid_photo_to_monet_B2A"], FM),
        ("KID std", "photo->Monet (B2A)", fm["kid_photo_to_monet_B2A_std"], FM),
        ("KID mean", "Monet->photo (A2B)", fm["kid_monet_to_photo_A2B"], FM),
        ("KID std", "Monet->photo (A2B)", fm["kid_monet_to_photo_A2B_std"], FM),
        ("precision", "photo->Monet (B2A)", fm["precision_photo_to_monet_B2A"], FM),
        ("recall", "photo->Monet (B2A)", fm["recall_photo_to_monet_B2A"], FM),
        ("precision", "Monet->photo (A2B)", fm["precision_monet_to_photo_A2B"], FM),
        ("recall", "Monet->photo (A2B)", fm["recall_monet_to_photo_A2B"], FM),
        ("cycle-reconstruction L1", "photo->Monet->photo", cyc_b["cycle_l1_mean"], FM + f" ({cyc_b['n_images']} images)"),
        ("cycle-reconstruction L1", "Monet->photo->Monet", cyc_a["cycle_l1_mean"], FM + f" ({cyc_a['n_images']} images)"),
        ("LPIPS input vs reconstruction", "photo->Monet->photo", cyc_b["lpips_mean"], FM),
        ("LPIPS input vs reconstruction", "Monet->photo->Monet", cyc_a["lpips_mean"], FM),
        ("content cosine input vs translation", "photo->Monet (B2A)", cyc_b["content_cosine_sim_mean"], FM),
        ("content cosine input vs translation", "Monet->photo (A2B)", cyc_a["content_cosine_sim_mean"], FM),
        ("generator / discriminator loss curves", "both", "outputs/loss_curves.png", "per-epoch values in " + TR),
        ("final generator loss", "both", tm["final_loss_G"], TR),
        ("final discriminator loss", "both", tm["final_loss_D"], TR),
        ("final cycle-consistency loss (weighted)", "both", tm["final_loss_cycle"], TR),
        ("final identity loss (weighted)", "both", tm["final_loss_identity"], TR),
        ("gradient norm mean (G + D)", "both", tm["mean_grad_norm"], TR),
        ("gradient norm max (G + D)", "both", tm["max_grad_norm"], TR),
        ("NaN / Inf count", "both", tm["nan_or_inf_events"], TR),
        ("parameter count (G_A2B + G_B2A + D_A + D_B)", "both", tm["total_parameter_count"], TR),
        ("training time (s)", "both", tm["total_training_time_sec"], TR),
        ("images per second (per domain)", "both", tm["avg_images_per_sec"], TR),
        ("peak GPU memory (MB)", "both", tm["peak_gpu_memory_mb"], TR),
        ("peak process memory (MB)", "both", tm["peak_memory_mb"], TR),
        ("epochs / steps", "both", f"{tm['epochs_trained']} / {tm['steps']}", TR),
        ("submitted checkpoint", "both", "checkpoints/best.pt", f"epoch {best['epoch']} {best['weights']} weights, "
                                                                 f"best of {len(scores)} scored in outputs/checkpoint_scores.csv"),
    ]

    audit_path = os.path.join(OUT, "human_audit", "human_audit_results.json")
    if os.path.exists(audit_path):
        for k, v in load(audit_path).items():
            rows.append((f"human audit: {k}", "photo->Monet (B2A)", v, "outputs/human_audit/human_audit_results.json"))
    else:
        rows.append(("human audit score / inter-rater agreement", "photo->Monet (B2A)", "pending",
                     "30 blinded samples prepared in outputs/human_audit/; run human_audit.py score after both raters"))

    out_path = os.path.join(MEMBER, "metrics_report.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "direction", "value", "source"])
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
