"""Local CycleGAN metrics for Lab 1 Part 3.

Run after training + inference (from member folder or repo root):
  python task3_gan/sneha_singh/evaluate_local.py

Computes FID / KID / precision-recall (torch-fidelity), LPIPS, content-cosine,
and a local MiFID estimate (Inception cosine distance vs target domain).
Writes:
  - full_metrics_report.csv  (report metrics)
  - submission.csv           (Kaggle upload: FID + MiFID from A2B / photo→Monet)

Class competition upload is submission.csv (self-reported scores), NOT images.zip.

Prefer the course evaluation script + task3_gan/data/real_stats.npz when available;
this file is a local stand-in until those match exactly.

Integrity note: Inception/VGG here are for *measurement only*, not image generation.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
    """Kaggle upload for this class competition: self-reported FID + MiFID only.

    Column names follow the lab PDF (FID/MiFID). If sample_submission.csv differs,
    rename columns to match that file exactly before upload.
    """
    with SUBMISSION_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fid", "mifid"])
        w.writeheader()
        w.writerow({"fid": fid, "mifid": mifid})
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


def _pair_paths(fake_dir: Path, real_dir: Path, limit: int) -> list[tuple[Path, Path]]:
    fakes = _list_images(fake_dir)
    reals = _list_images(real_dir)
    n = min(len(fakes), len(reals), limit)
    return list(zip(fakes[:n], reals[:n]))


@torch.no_grad()
def _mifid(
    fake_dir: Path, real_dir: Path, device: torch.device, n_sample: int = 200
) -> float | str:
    """Local MiFID estimate: avg cosine *distance* of Inception-v3 features.

    Compares generated images to real *target-domain* images (not source photos).
    Pairing after equal-size subsample approximates the course description.
    """
    fakes = _list_images(fake_dir)
    reals = _list_images(real_dir)
    if len(fakes) < 2 or len(reals) < 2:
        return ""
    n = min(len(fakes), len(reals), n_sample)
    rng = random.Random(670)
    fakes = rng.sample(fakes, n)
    reals = rng.sample(reals, n)

    try:
        inc = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    except Exception:
        inc = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inc.fc = torch.nn.Identity()
    inc = inc.to(device).eval()

    tf = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    def feats(paths: list[Path]) -> torch.Tensor:
        out = []
        for p in paths:
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            y = inc(x)
            if isinstance(y, tuple):
                y = y[0]
            out.append(y.squeeze(0))
        return torch.stack(out)

    f_fake, f_real = feats(fakes), feats(reals)
    cos = F.cosine_similarity(f_fake, f_real, dim=1)
    return float((1.0 - cos).mean().item())


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
    print("pred_A2B:", len(_list_images(pred_a2b)), "| monet:", len(_list_images(monet)))
    print("pred_B2A:", len(_list_images(pred_b2a)), "| photo:", len(_list_images(photo)))
    _probe_real_stats()

    existing = _load_existing()
    limit = 64
    mifid_n = 200
    if CFG_PATH.exists():
        cfg = json.loads(CFG_PATH.read_text())
        if cfg.get("smoke"):
            limit = 16
            mifid_n = 16

    def _row(direction, fid_block, mifid, lp, cos):
        base = {k: "" for k in FIELDS}
        base.update(existing.get(direction, {}))
        base["direction"] = direction
        base.update(fid_block)
        base["mifid"] = mifid
        base["lpips"] = lp
        base["content_cosine"] = cos
        try:
            base["leaderboard_proxy"] = (float(base["fid"]) + float(mifid)) / 2.0
        except (TypeError, ValueError):
            base["leaderboard_proxy"] = ""
        return base

    # --- A2B = Kaggle direction (photo→Monet); save as soon as ready ---
    print("computing A2B FID/KID/PR ...", flush=True)
    a2b_fid = _fidelity_metrics(pred_a2b, monet, dev_str)
    print("computing A2B MiFID (generated Monet vs real Monet) ...", flush=True)
    a2b_mifid = _mifid(pred_a2b, monet, device, n_sample=mifid_n)
    a2b_pairs = _pair_paths(pred_a2b, photo, limit)
    print("computing A2B LPIPS/content on", len(a2b_pairs), "pairs ...", flush=True)
    a2b_lpips, a2b_cos = _lpips_and_content(a2b_pairs, device, max_n=limit)

    a2b_row = _row("A2B", a2b_fid, a2b_mifid, a2b_lpips, a2b_cos)
    _write_submission(a2b_row.get("fid", ""), a2b_row.get("mifid", ""))
    print(
        "A2B ready →",
        "fid=", a2b_row["fid"],
        "mifid=", a2b_row["mifid"],
        "proxy=", a2b_row["leaderboard_proxy"],
        flush=True,
    )

    # --- B2A report-only; skip heavy FID if only smoke preds (was hanging vs 7k photos) ---
    n_b2a = len(_list_images(pred_b2a))
    if n_b2a < 50:
        print(
            f"skipping B2A FID/KID/PR (only {n_b2a} preds; need full Monet→photo set)",
            flush=True,
        )
        b2a_fid = {"fid": "", "kid": "", "precision": "", "recall": ""}
    else:
        print("computing B2A FID/KID/PR ...", flush=True)
        b2a_fid = _fidelity_metrics(pred_b2a, photo, dev_str, min_fake=50)
    print("computing B2A MiFID (generated photo vs real photo) ...", flush=True)
    b2a_mifid = _mifid(pred_b2a, photo, device, n_sample=min(mifid_n, max(2, n_b2a)))
    b2a_pairs = _pair_paths(pred_b2a, monet, limit)
    print("computing B2A LPIPS/content on", len(b2a_pairs), "pairs ...", flush=True)
    b2a_lpips, b2a_cos = _lpips_and_content(b2a_pairs, device, max_n=limit)
    b2a_row = _row("B2A", b2a_fid, b2a_mifid, b2a_lpips, b2a_cos)

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
        "Note: numbers are local estimates until you run the course evaluation "
        "script (and real_stats.npz). Match sample_submission.csv column names if different.",
        flush=True,
    )


if __name__ == "__main__":
    main()
