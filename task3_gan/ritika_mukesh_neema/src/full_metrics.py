"""
Task 3.2 - Evaluation and Analysis: every metric except the official FID / MiFID
(those come from ../evaluate_local.py, the TA's script).

Computes every metric in the assignment's Task 3 metrics list that can be
computed locally (both directions where applicable):
  FID, KID, generative precision/recall (kNN manifold estimate), cycle-
  reconstruction L1 distance, LPIPS, content-preservation cosine similarity,
  gradient norms/NaN count + gen/disc/cycle/identity loss values (pulled from
  train_metrics.json), parameter count/training time/images-per-sec/peak
  memory (also pulled from train_metrics.json).
Kaggle leaderboard score and the human audit are recorded separately (see
README / human_audit.py) since they need an external submission / human
raters respectively.

Uses torchvision's pretrained Inception v3 pool features for FID/KID/
precision-recall (standard practice -- these metrics are defined in terms of
the Inception feature space) and the `lpips` package for perceptual
similarity. Both need one-time internet access to fetch pretrained weights.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from PIL import Image
import torchvision.transforms as T
import torchvision.models as tvm

from inference import load_generators
from dataset import list_images
from utils import denorm


# ---------------------------------------------------------------- features --
def get_inception_feature_extractor(device):
    weights = tvm.Inception_V3_Weights.IMAGENET1K_V1
    model = tvm.inception_v3(weights=weights, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    preprocess = T.Compose([
        T.Resize((299, 299), Image.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


@torch.no_grad()
def extract_features(folder, model, preprocess, device, max_images=None, batch_size=32):
    files = list_images(folder)
    if max_images:
        files = files[:max_images]
    feats = []
    batch = []
    for path in files:
        img = preprocess(Image.open(path).convert("RGB"))
        batch.append(img)
        if len(batch) == batch_size:
            x = torch.stack(batch).to(device)
            feats.append(model(x).cpu().numpy())
            batch = []
    if batch:
        x = torch.stack(batch).to(device)
        feats.append(model(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


# --------------------------------------------------------------------- FID --
def compute_fid(feat_real, feat_fake):
    mu1, sigma1 = feat_real.mean(axis=0), np.cov(feat_real, rowvar=False)
    mu2, sigma2 = feat_fake.mean(axis=0), np.cov(feat_fake, rowvar=False)
    diff = mu1 - mu2
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


# --------------------------------------------------------------------- KID --
def compute_kid(feat_real, feat_fake, n_subsets=50, subset_size=None, degree=3, coef0=1.0, seed=42):
    rng = np.random.default_rng(seed)
    n = min(len(feat_real), len(feat_fake))
    subset_size = subset_size or min(100, n)
    d = feat_real.shape[1]

    def poly_kernel(x, y):
        return (x @ y.T / d + coef0) ** degree

    scores = []
    for _ in range(n_subsets):
        idx_r = rng.choice(len(feat_real), subset_size, replace=False)
        idx_f = rng.choice(len(feat_fake), subset_size, replace=False)
        x, y = feat_real[idx_r], feat_fake[idx_f]
        m = subset_size
        Kxx = poly_kernel(x, x)
        Kyy = poly_kernel(y, y)
        Kxy = poly_kernel(x, y)
        kid = (
            (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
            - 2 * Kxy.mean()
        )
        scores.append(kid)
    return float(np.mean(scores)), float(np.std(scores))


# ----------------------------------------------------- precision / recall --
def _pairwise_dist(a, b):
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def compute_precision_recall(feat_real, feat_fake, k=3, max_samples=500, seed=42):
    """Improved precision/recall (Kynkaanniemi et al. 2019), kNN manifold estimate."""
    rng = np.random.default_rng(seed)
    if len(feat_real) > max_samples:
        feat_real = feat_real[rng.choice(len(feat_real), max_samples, replace=False)]
    if len(feat_fake) > max_samples:
        feat_fake = feat_fake[rng.choice(len(feat_fake), max_samples, replace=False)]

    d_rr = _pairwise_dist(feat_real, feat_real)
    np.fill_diagonal(d_rr, np.inf)
    radii_real = np.sort(d_rr, axis=1)[:, k - 1]

    d_ff = _pairwise_dist(feat_fake, feat_fake)
    np.fill_diagonal(d_ff, np.inf)
    radii_fake = np.sort(d_ff, axis=1)[:, k - 1]

    d_rf = _pairwise_dist(feat_real, feat_fake)  # rows=real, cols=fake

    # precision: fraction of fake samples inside at least one real sample's kNN ball
    within_real_ball = (d_rf <= radii_real[:, None]).any(axis=0)
    precision = float(within_real_ball.mean())

    # recall: fraction of real samples inside at least one fake sample's kNN ball
    within_fake_ball = (d_rf.T <= radii_fake[:, None]).any(axis=0)
    recall = float(within_fake_ball.mean())

    return precision, recall


# --------------------------------------------------- cycle-recon / LPIPS ---
@torch.no_grad()
def cycle_reconstruction_and_lpips(G_A2B, G_B2A, folder, device, preprocess_256, lpips_model,
                                    max_images=100, direction="A"):
    files = list_images(folder)[:max_images]
    l1_scores, lpips_scores, cosine_scores = [], [], []
    for path in files:
        img = preprocess_256(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        if direction == "A":  # monet -> photo -> monet
            fake = G_A2B(img)
            rec = G_B2A(fake)
        else:  # photo -> monet -> photo
            fake = G_B2A(img)
            rec = G_A2B(fake)
        l1_scores.append(F.l1_loss(rec, img).item())
        if lpips_model is not None:
            lpips_scores.append(lpips_model(rec, img).mean().item())
        # content-preservation: cosine sim between input and its single-pass translation
        # (flattened pixel space, cheap proxy that doesn't need another network call)
        cos = F.cosine_similarity(img.flatten(1), fake.flatten(1)).item()
        cosine_scores.append(cos)
    return {
        "cycle_l1_mean": float(np.mean(l1_scores)),
        "lpips_mean": float(np.mean(lpips_scores)) if lpips_scores else None,
        "content_cosine_sim_mean": float(np.mean(cosine_scores)),
        "n_images": len(files),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../checkpoints/best.pt")
    ap.add_argument("--data_dir_a", default="../../data/monet_jpg")
    ap.add_argument("--data_dir_b", default="../../data/photo_jpg")
    ap.add_argument("--out_dir", default="../outputs")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--n_blocks", type=int, default=None, help="default: from the checkpoint")
    ap.add_argument("--max_images_for_fid", type=int, default=300)
    ap.add_argument("--max_images_for_cycle", type=int, default=100)
    ap.add_argument("--skip_lpips", action="store_true", help="skip LPIPS if the package/weights aren't available")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_a2b_dir = os.path.join(args.out_dir, "pred_A2B")
    pred_b2a_dir = os.path.join(args.out_dir, "pred_B2A")
    for d in [pred_a2b_dir, pred_b2a_dir]:
        if not os.path.isdir(d) or len(list_images(d)) == 0:
            raise SystemExit(f"{d} is empty -- run generate.py first.")

    print("Loading Inception v3 feature extractor (downloads pretrained weights on first run)...")
    inception, incep_preprocess = get_inception_feature_extractor(device)

    print("Extracting features: real monet (A) / real photo (B) / fake monet (pred_B2A) / fake photo (pred_A2B)...")
    feat_real_A = extract_features(args.data_dir_a, inception, incep_preprocess, device, args.max_images_for_fid)
    feat_real_B = extract_features(args.data_dir_b, inception, incep_preprocess, device, args.max_images_for_fid)
    feat_fake_A = extract_features(pred_b2a_dir, inception, incep_preprocess, device, args.max_images_for_fid)
    feat_fake_B = extract_features(pred_a2b_dir, inception, incep_preprocess, device, args.max_images_for_fid)

    print("Computing FID / KID / precision-recall (both directions)...")
    fid_B2A = compute_fid(feat_real_A, feat_fake_A)  # fake monet vs real monet -- Kaggle-scored direction
    fid_A2B = compute_fid(feat_real_B, feat_fake_B)  # fake photo vs real photo
    kid_B2A, kid_B2A_std = compute_kid(feat_real_A, feat_fake_A)
    kid_A2B, kid_A2B_std = compute_kid(feat_real_B, feat_fake_B)
    prec_B2A, rec_B2A = compute_precision_recall(feat_real_A, feat_fake_A)
    prec_A2B, rec_A2B = compute_precision_recall(feat_real_B, feat_fake_B)

    print("Loading generators for cycle-reconstruction / LPIPS / content-cosine metrics...")
    # architecture (filters / blocks / upsampling) comes from the checkpoint's own config
    G_A2B, G_B2A, _, _ = load_generators(args.ckpt, device, n_blocks=args.n_blocks)

    lpips_model = None
    if not args.skip_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex").to(device)
            lpips_model.eval()
        except Exception as e:
            print(f"  [warn] could not load LPIPS ({e}); continuing without it (pass --skip_lpips to silence this).")

    cycle_transform = T.Compose([
        T.Resize((args.img_size, args.img_size), Image.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    cyc_A = cycle_reconstruction_and_lpips(
        G_A2B, G_B2A, args.data_dir_a, device, cycle_transform, lpips_model,
        args.max_images_for_cycle, direction="A",
    )
    cyc_B = cycle_reconstruction_and_lpips(
        G_A2B, G_B2A, args.data_dir_b, device, cycle_transform, lpips_model,
        args.max_images_for_cycle, direction="B",
    )

    # pull training-side metrics (loss values, grad norms, params, timing, memory)
    train_metrics_path = os.path.join(args.out_dir, "train_metrics.json")
    train_metrics = {}
    if os.path.exists(train_metrics_path):
        with open(train_metrics_path) as f:
            train_metrics = json.load(f)
            train_metrics.pop("loss_history", None)

    report = {
        "fid_photo_to_monet_B2A": fid_B2A,
        "fid_monet_to_photo_A2B": fid_A2B,
        "kid_photo_to_monet_B2A": kid_B2A, "kid_photo_to_monet_B2A_std": kid_B2A_std,
        "kid_monet_to_photo_A2B": kid_A2B, "kid_monet_to_photo_A2B_std": kid_A2B_std,
        "precision_photo_to_monet_B2A": prec_B2A, "recall_photo_to_monet_B2A": rec_B2A,
        "precision_monet_to_photo_A2B": prec_A2B, "recall_monet_to_photo_A2B": rec_A2B,
        "cycle_A_monet_photo_monet": cyc_A,
        "cycle_B_photo_monet_photo": cyc_B,
        **train_metrics,
    }

    with open(os.path.join(args.out_dir, "full_metrics_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    # repo structure: <member>/src/ also holds the full metrics file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    flat_row = {}
    for k, v in report.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat_row[f"{k}.{kk}"] = vv
        else:
            flat_row[k] = v
    csv_path = os.path.join(os.path.dirname(args.out_dir) or ".", "full_metrics_report.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in flat_row.items():
            w.writerow([k, v])

    print(f"\nWrote {os.path.join(args.out_dir, 'full_metrics_report.json')} and {csv_path}")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
