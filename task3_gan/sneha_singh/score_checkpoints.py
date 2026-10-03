"""Score several CycleGAN checkpoints with the professor evaluation and combine them in one CSV.

For each checkpoint: translate the first 300 sorted Monet images (A2B, Monet -> photo) and the
first 300 sorted photos (B2A, photo -> Monet), then compute FID and MiFID exactly as the course
notebook does (evaluate_local.calculate_fid_mifid). Score = (mean FID + mean MiFID) / 2.

Usage (from the repo root):
  python task3_gan/sneha_singh/score_checkpoints.py \
      --ckpts path/best_epoch_80.pt path/best_epoch_90.pt \
      --upsample nearest --ngf 64 \
      --out task3_gan/sneha_singh/outputs/checkpoint_scores.csv

--upsample and --ngf must match the run that produced the checkpoints (the weights load either way,
but the wrong mode gives wrong images and a much worse score).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

MEMBER = Path(__file__).resolve().parent
DATA = MEMBER.parent / "data"
N_EVAL = 300
IMG = 256

spec = importlib.util.spec_from_file_location("evaluate_local", MEMBER / "evaluate_local.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """Same generator as src/part3_cyclegan.ipynb; only the upsample mode is a parameter."""

    def __init__(self, upsample, ngf=64, n_blocks=9):
        super().__init__()
        up = dict(mode="nearest") if upsample == "nearest" else dict(mode="bilinear", align_corners=False)
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf, affine=False, track_running_stats=False),
            nn.ReLU(True),
        ]
        n = ngf
        for _ in range(2):
            model += [
                nn.Conv2d(n, n * 2, 3, 2, 1, bias=False),
                nn.InstanceNorm2d(n * 2, affine=False, track_running_stats=False),
                nn.ReLU(True),
            ]
            n *= 2
        model += [ResnetBlock(n) for _ in range(n_blocks)]
        for _ in range(2):
            model += [
                nn.Upsample(scale_factor=2, **up),
                nn.ReflectionPad2d(1),
                nn.Conv2d(n, n // 2, 3, stride=1, padding=0, bias=False),
                nn.InstanceNorm2d(n // 2, affine=False, track_running_stats=False),
                nn.ReLU(True),
            ]
            n //= 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# same eval transform and JPEG saving as translate_folder in the notebook
TF = transforms.Compose([
    transforms.Resize((IMG, IMG), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


@torch.no_grad()
def translate(gen, paths, out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(paths):
        y = gen(TF(Image.open(p).convert("RGB")).unsqueeze(0).to(device))[0].cpu().clamp(-1, 1)
        arr = ((y + 1) * 0.5).permute(1, 2, 0).numpy() * 255
        Image.fromarray(arr.astype(np.uint8)).save(out_dir / ("%05d.jpg" % i), quality=95)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--upsample", choices=["nearest", "bilinear"], required=True)
    ap.add_argument("--ngf", type=int, default=64, help="generator base width (config ngf)")
    ap.add_argument("--run_id", default="")
    ap.add_argument("--out", default=str(MEMBER / "outputs" / "checkpoint_scores.csv"))
    args = ap.parse_args()

    device = pick_device()
    monet = ev._list_images(DATA / "monet_jpg")[:N_EVAL]
    photo = ev._list_images(DATA / "photo_jpg")[:N_EVAL]
    assert monet and photo, "missing task3_gan/data/monet_jpg or photo_jpg"
    print("device:", device, "| monet:", len(monet), "| photo:", len(photo), "| upsample:", args.upsample)

    G_AB = ResnetGenerator(args.upsample, ngf=args.ngf).to(device).eval()  # photo -> Monet
    G_BA = ResnetGenerator(args.upsample, ngf=args.ngf).to(device).eval()  # Monet -> photo
    rows = []
    for ck in args.ckpts:
        st = torch.load(ck, map_location=device, weights_only=True)
        G_AB.load_state_dict(st["G_AB"])
        G_BA.load_state_dict(st["G_BA"])
        with tempfile.TemporaryDirectory() as tmp:
            a2b, b2a = Path(tmp) / "pred_A2B", Path(tmp) / "pred_B2A"
            translate(G_BA, monet, a2b, device)
            translate(G_AB, photo, b2a, device)
            fid_b2a, mifid_b2a = ev.calculate_fid_mifid(monet, ev._list_images(b2a)[:N_EVAL], device)
            fid_a2b, mifid_a2b = ev.calculate_fid_mifid(photo, ev._list_images(a2b)[:N_EVAL], device)
        fid, mifid = (fid_a2b + fid_b2a) / 2, (mifid_a2b + mifid_b2a) / 2
        row = {
            "run_id": args.run_id,
            "checkpoint": Path(ck).name,
            "epoch": st.get("epoch", ""),
            "upsample": args.upsample,
            "ngf": args.ngf,
            "fid_b2a": round(fid_b2a, 4), "mifid_b2a": round(mifid_b2a, 4),
            "fid_a2b": round(fid_a2b, 4), "mifid_a2b": round(mifid_a2b, 4),
            "submission_fid": round(fid, 4), "submission_mifid": round(mifid, 4),
            "score": round((fid + mifid) / 2, 4),
        }
        rows.append(row)
        print(row, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["score"]))
    best = min(rows, key=lambda r: r["score"])
    print("wrote", out, "| best:", best["checkpoint"], "score", best["score"])


if __name__ == "__main__":
    main()
