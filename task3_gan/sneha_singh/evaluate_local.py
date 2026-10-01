"""Local CycleGAN metrics for Lab 1 Part 3.

Run after training + inference (from member folder or repo root):
  python task3_gan/sneha_singh/evaluate_local.py

FID and MiFID follow the course notebook: Inception-v3, at most 300 sorted
images, then cosine distance paired by index.
submission.csv is the average of both directions:
  FID = (photo→Monet FID + Monet→photo FID) / 2
  MiFID = (photo→Monet MiFID + Monet→photo MiFID) / 2
Folders match that notebook:
  pred_A2B = Monet → photo (generated photos)
  pred_B2A = photo → Monet (generated Monet)
full_metrics_report.csv still keeps both directions plus the other lab metrics.

Integrity note: Inception/VGG here are for measurement only, not image generation.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.distance import cosine
from torchvision import models, transforms

MEMBER = Path(__file__).resolve().parent
TASK = MEMBER.parent
REPO = TASK.parent
DATA = TASK / "data"
OUT = MEMBER / "outputs"
CSV_PATH = MEMBER / "full_metrics_report.csv"
SUBMISSION_PATH = MEMBER / "submission.csv"
REAL_STATS = DATA / "real_stats.npz"
CFG_PATH = MEMBER / "src" / "config.json"
N_EVAL = 300
FID_BATCH = 32

FIELDS = [
    "direction",
    "fid",
    "kid",
    "mifid",
    "precision",
    "recall",
    "cycle_l1",
    "lpips",
    "content_cosine",
    "g_loss",
    "d_loss",
    "cycle_loss",
    "identity_loss",
    "grad_norm",
    "nan_count",
    "param_count",
    "train_time_sec",
    "images_per_sec",
    "peak_memory_mb",
    "human_audit_score",
    "inter_rater_agreement",
    "kaggle_public",
    "kaggle_private",
    "kaggle_rank",
    "leaderboard_proxy",  # (fid + mifid) / 2 when both available
]


def _list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def _probe_real_stats() -> None:
    """Print keys if course real_stats.npz is present (wire exact FID/MiFID later)."""
    if not REAL_STATS.exists():
        print(
            "missing",
            REAL_STATS.relative_to(REPO),
            "— download from Kaggle Data tab; course eval should load it",
        )
        return
    try:
        z = np.load(REAL_STATS, allow_pickle=False)
        print("found", REAL_STATS.relative_to(REPO), "keys:", list(z.keys()))
        z.close()
    except Exception as e:
        print("could not read", REAL_STATS, ":", e)


def _write_submission(fid, mifid) -> None:
    """Kaggle upload: submission.csv with header ID,FID,MiFID and one data row."""
    with SUBMISSION_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "FID", "MiFID"])
        w.writeheader()
        w.writerow({"ID": 1, "FID": fid, "MiFID": mifid})
    print("wrote", SUBMISSION_PATH.relative_to(REPO), "← upload this to Kaggle")


def _load_existing() -> dict[str, dict]:
    if not CSV_PATH.exists():
        return {}
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    return {r.get("direction", ""): r for r in rows if r.get("direction")}


def _fidelity_metrics(
    fake_dir: Path, real_dir: Path, device: str, *, min_fake: int = 2
) -> dict:
    """FID / KID / precision / recall via torch-fidelity."""
    out = {"fid": "", "kid": "", "precision": "", "recall": ""}
    try:
        from torch_fidelity import calculate_metrics
    except ImportError:
        print("torch-fidelity not installed — skip FID/KID/PR (pip install torch-fidelity)")
        return out

    n_fake, n_real = len(_list_images(fake_dir)), len(_list_images(real_dir))
    if n_fake < min_fake or n_real < 2:
        print(
            "skip FID/KID/PR — need >=",
            min_fake,
            "fakes and >=2 reals; got",
            n_fake,
            n_real,
        )
        return out

    # torch-fidelity defaults kid_subset_size=1000; Monet set is only ~300
    kid_n = max(2, min(100, n_fake, n_real))
    print(f"  FID inputs: {n_fake} fake vs {n_real} real (kid_subset={kid_n})", flush=True)
    metrics = calculate_metrics(
        input1=str(fake_dir),
        input2=str(real_dir),
        cuda=(device == "cuda"),
        fid=True,
        kid=True,
        kid_subset_size=kid_n,
        prc=True,
        verbose=False,
    )
    out["fid"] = metrics.get("frechet_inception_distance", "")
    out["kid"] = metrics.get("kernel_inception_distance_mean", metrics.get("kernel_inception_distance", ""))
    out["precision"] = metrics.get("precision", "")
    out["recall"] = metrics.get("recall", "")
    return out


def _take_n(paths: list[Path], n: int | None) -> list[Path]:
    """Course script: sorted paths, then the first n."""
    if n is None:
        return paths
    return paths[: min(n, len(paths))]


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    product = sigma1.dot(sigma2)
    try:
        covmean = scipy.linalg.sqrtm(product, disp=False)[0]
    except TypeError:
        covmean = scipy.linalg.sqrtm(product)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        product = (sigma1 + offset).dot(sigma2 + offset)
        try:
            covmean = scipy.linalg.sqrtm(product, disp=False)[0]
        except TypeError:
            covmean = scipy.linalg.sqrtm(product)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def _inception(device: torch.device) -> nn.Module:
    model = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False,
    )
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model


_INCEPTION_TF = transforms.Compose(
    [
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


@torch.no_grad()
def _activations(model: nn.Module, paths: list[Path], device: torch.device) -> np.ndarray:
    feats = []
    for i in range(0, len(paths), FID_BATCH):
        batch = []
        for p in paths[i : i + FID_BATCH]:
            batch.append(_INCEPTION_TF(Image.open(p).convert("RGB")))
        x = torch.stack(batch, dim=0).to(device)
        y = model(x)
        if isinstance(y, tuple):
            y = y[0]
        feats.append(y.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def calculate_fid_mifid(
    real_paths: list[Path], gen_paths: list[Path], device: torch.device
) -> tuple[float | str, float | str]:
    """Same FID and MiFID as the course notebook."""
    real_paths = sorted(real_paths)
    gen_paths = sorted(gen_paths)
    if len(real_paths) < 2 or len(gen_paths) < 2:
        return "", ""
    n = min(len(real_paths), len(gen_paths))
    real_paths, gen_paths = real_paths[:n], gen_paths[:n]
    print(f"  FID/MiFID: {n} real vs {n} generated", flush=True)
    model = _inception(device)
    real_act = _activations(model, real_paths, device)
    gen_act = _activations(model, gen_paths, device)
    mu_r, sig_r = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
    mu_g, sig_g = gen_act.mean(axis=0), np.cov(gen_act, rowvar=False)
    fid = _frechet_distance(mu_r, sig_r, mu_g, sig_g)
    m = min(len(real_act), len(gen_act))
    mifid = float(np.mean([cosine(real_act[i], gen_act[i]) for i in range(m)]))
    return fid, mifid


def _pair_paths(fake_dir: Path, real_dir: Path, limit: int) -> list[tuple[Path, Path]]:
    fakes = _list_images(fake_dir)
    reals = _list_images(real_dir)
    n = min(len(fakes), len(reals), limit)
    return list(zip(fakes[:n], reals[:n]))


@torch.no_grad()
def _lpips_and_content(
    pairs: list[tuple[Path, Path]], device: torch.device, max_n: int = 64
) -> tuple[float | str, float | str]:
    if not pairs:
        return "", ""
    pairs = pairs[:max_n]
    tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    lpips_val: float | str = ""
    try:
        import lpips

        loss_fn = lpips.LPIPS(net="alex").to(device)
        scores = []
        for fa, re in pairs:
            t0 = tf(Image.open(fa).convert("RGB")).unsqueeze(0).to(device)
            t1 = tf(Image.open(re).convert("RGB")).unsqueeze(0).to(device)
            scores.append(float(loss_fn(t0, t1).item()))
        lpips_val = float(np.mean(scores))
    except ImportError:
        print("lpips not installed — skip LPIPS (pip install lpips)")

    # content cosine: generated vs *source* photo (report metric — NOT Kaggle MiFID)
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].to(device).eval()
    cos_scores = []
    for fa, re in pairs:
        t0 = tf(Image.open(fa).convert("RGB")).unsqueeze(0).to(device)
        t1 = tf(Image.open(re).convert("RGB")).unsqueeze(0).to(device)
        f0 = vgg(t0).flatten(1)
        f1 = vgg(t1).flatten(1)
        cos_scores.append(float(F.cosine_similarity(f0, f1).item()))
    content = float(np.mean(cos_scores)) if cos_scores else ""
    return lpips_val, content


def main() -> None:
    random.seed(670)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dev_str = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        dev_str = "cpu"
    else:
        device = torch.device("cpu")
        dev_str = "cpu"

    pred_a2b = OUT / "pred_A2B"
    pred_b2a = OUT / "pred_B2A"
    monet = DATA / "monet_jpg"
    photo = DATA / "photo_jpg"

    print("member:", MEMBER.name)
    print("device:", device, "| fidelity backend:", dev_str)
    n_a2b = len(_list_images(pred_a2b))
    n_b2a = len(_list_images(pred_b2a))
    print("pred_A2B:", n_a2b, "| pred_B2A:", n_b2a)
    print("real photo:", len(_list_images(photo)), "| real Monet:", len(_list_images(monet)))
    # Professor folders: pred_A2B = Monet→photo, pred_B2A = photo→Monet.
    # An older notebook wrote those two sets the other way around (7038 in pred_A2B).
    old_layout = n_a2b > n_b2a and n_a2b >= 1000
    if old_layout:
        print("older folder layout: pred_A2B is photo→Monet, pred_B2A is Monet→photo")
        gen_photo_dir, gen_monet_dir = pred_b2a, pred_a2b
    else:
        gen_photo_dir, gen_monet_dir = pred_a2b, pred_b2a

    existing = _load_existing()
    limit = 64
    n_eval = N_EVAL
    if CFG_PATH.exists():
        cfg = json.loads(CFG_PATH.read_text())
        if cfg.get("smoke"):
            limit = 16
            n_eval = 16

    def _row(direction, fid_block, mifid, lp, cos):
        base = {k: "" for k in FIELDS}
        base.update(existing.get(direction, {}))
        base["direction"] = direction
        for key, value in fid_block.items():
            if value != "":
                base[key] = value
        if mifid != "":
            base["mifid"] = mifid
        if lp != "":
            base["lpips"] = lp
        if cos != "":
            base["content_cosine"] = cos
        try:
            base["leaderboard_proxy"] = (float(base["fid"]) + float(mifid)) / 2.0
        except (TypeError, ValueError):
            base["leaderboard_proxy"] = ""
        return base

    # A2B: generated photos vs real photos. B2A: generated Monet vs real Monet.
    real_monet = _take_n(_list_images(monet), n_eval)
    real_photo = _take_n(_list_images(photo), n_eval)
    gen_photo = _take_n(_list_images(gen_photo_dir), n_eval)
    gen_monet = _take_n(_list_images(gen_monet_dir), n_eval)
    print("\nEvaluating Photo -> Monet (B2A)", flush=True)
    fid_b2a, mifid_b2a = calculate_fid_mifid(real_monet, gen_monet, device)
    print(f"[Photo->Monet] FID={fid_b2a}  MiFID={mifid_b2a}", flush=True)
    print("computing B2A KID/PR ...", flush=True)
    b2a_fid = _fidelity_metrics(gen_monet_dir, monet, dev_str)
    b2a_fid["fid"] = fid_b2a
    b2a_mifid = mifid_b2a
    b2a_pairs = _pair_paths(gen_monet_dir, photo, limit)
    print("computing B2A LPIPS/content on", len(b2a_pairs), "pairs ...", flush=True)
    b2a_lpips, b2a_cos = _lpips_and_content(b2a_pairs, device, max_n=limit)
    b2a_row = _row("B2A", b2a_fid, b2a_mifid, b2a_lpips, b2a_cos)

    print("\nEvaluating Monet -> Photo (A2B)", flush=True)
    fid_a2b, mifid_a2b = calculate_fid_mifid(real_photo, gen_photo, device)
    print(f"[Monet->Photo] FID={fid_a2b}  MiFID={mifid_a2b}", flush=True)
    n_gen_photo = len(_list_images(gen_photo_dir))
    if n_gen_photo < 50:
        print(
            f"skipping A2B KID/PR (only {n_gen_photo} preds)",
            flush=True,
        )
        a2b_fid = {"fid": "", "kid": "", "precision": "", "recall": ""}
    else:
        print("computing A2B KID/PR ...", flush=True)
        a2b_fid = _fidelity_metrics(gen_photo_dir, photo, dev_str, min_fake=50)
    a2b_fid["fid"] = fid_a2b
    a2b_mifid = mifid_a2b
    a2b_pairs = _pair_paths(gen_photo_dir, monet, limit)
    print("computing A2B LPIPS/content on", len(a2b_pairs), "pairs ...", flush=True)
    a2b_lpips, a2b_cos = _lpips_and_content(a2b_pairs, device, max_n=limit)
    a2b_row = _row("A2B", a2b_fid, a2b_mifid, a2b_lpips, a2b_cos)

    try:
        sub_fid = (float(fid_a2b) + float(fid_b2a)) / 2.0
        sub_mifid = (float(mifid_a2b) + float(mifid_b2a)) / 2.0
    except (TypeError, ValueError):
        sub_fid, sub_mifid = "", ""
    _write_submission(sub_fid, sub_mifid)
    print(
        "submission average of both directions →",
        "fid=", sub_fid,
        "mifid=", sub_mifid,
        flush=True,
    )

    rows = [a2b_row, b2a_row]
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print("wrote", CSV_PATH.relative_to(REPO), flush=True)
    for r in rows:
        print(
            r["direction"],
            "fid=", r["fid"],
            "mifid=", r["mifid"],
            "proxy=(FID+MiFID)/2=", r["leaderboard_proxy"],
            "kid=", r["kid"],
            "lpips=", r["lpips"],
            "content_cos=", r["content_cosine"],
            flush=True,
        )
    print(
        "submission.csv is the average of both directions, matching the course script.",
        flush=True,
    )


if __name__ == "__main__":
    main()
