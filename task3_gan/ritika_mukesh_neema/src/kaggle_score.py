"""
Compute this competition's official FID / MiFID against the provided
real_stats.npz (precomputed real-Monet Inception stats + 300 real feature
vectors), using the SAME Inception-v3 feature extractor as full_metrics.py,
and write the Kaggle submission CSV (ID,FID,MiFID).
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from scipy import linalg

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from full_metrics import get_inception_feature_extractor, extract_features  # noqa: E402


def compute_fid_from_stats(mu1, sigma1, feat_fake):
    mu2, sigma2 = feat_fake.mean(axis=0), np.cov(feat_fake, rowvar=False)
    diff = mu1.astype(np.float64) - mu2.astype(np.float64)
    try:
        covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    except TypeError:
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_mifid(feat_real, feat_fake, seed=42):
    rng = np.random.default_rng(seed)
    n = min(len(feat_real), len(feat_fake))
    real_idx = rng.choice(len(feat_real), n, replace=False) if len(feat_real) > n else np.arange(len(feat_real))
    fake_idx = rng.choice(len(feat_fake), n, replace=False) if len(feat_fake) > n else np.arange(len(feat_fake))
    r = feat_real[real_idx]
    f = feat_fake[fake_idx]
    r_norm = r / np.linalg.norm(r, axis=1, keepdims=True)
    f_norm = f / np.linalg.norm(f, axis=1, keepdims=True)
    cos_sim = f_norm @ r_norm.T  # (n, n) all-pairs cosine similarity
    d = 1.0 - cos_sim
    return float(d.mean()), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="../outputs/pred_B2A")
    ap.add_argument("--real_stats", default="../../data/real_stats.npz")
    ap.add_argument("--out_csv", default="../kaggle_submission.csv")
    ap.add_argument("--max_images_for_fid", type=int, default=None, help="None = use all generated images")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    stats = np.load(args.real_stats)
    mu_real, sigma_real, feats_real = stats["mu_real"], stats["sigma_real"], stats["feats_real"]
    print(f"Loaded real_stats.npz: mu_real {mu_real.shape}, sigma_real {sigma_real.shape}, feats_real {feats_real.shape}")

    print("Loading Inception v3 feature extractor (same as full_metrics.py)...")
    inception, preprocess = get_inception_feature_extractor(device)

    print(f"Extracting features from {args.pred_dir} ...")
    feat_fake = extract_features(args.pred_dir, inception, preprocess, device, args.max_images_for_fid)
    print(f"  -> {feat_fake.shape[0]} generated images, feature dim {feat_fake.shape[1]}")

    print("Computing FID against official real_stats (mu_real, sigma_real)...")
    fid = compute_fid_from_stats(mu_real, sigma_real, feat_fake)
    print(f"  FID = {fid}")

    print("Computing MiFID (mean cosine distance, real feats_real vs subsampled generated feats)...")
    mifid, n_used = compute_mifid(feats_real, feat_fake)
    print(f"  MiFID = {mifid} (n={n_used} per side)")

    with open(args.out_csv, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["ID", "FID", "MiFID"])
        w.writerow([1, fid, mifid])
    print(f"\nWrote {args.out_csv}")
    print(f"ID,FID,MiFID\n1,{fid},{mifid}")


if __name__ == "__main__":
    main()
