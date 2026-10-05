"""Regenerate the submitted Part 3 outputs from checkpoints/best.pt (epoch 91; best_generators.pt has the same generators).

- pred_A2B (Monet -> photo, 300 images): generator output averaged over 6 views
  (original, horizontal flip, and four 4-pixel shifts, each mapped back before averaging)
- pred_B2A (photo -> Monet, 7,038 images): single pass
- outputs/samples/grid.png: 4 photos, their Monet version, 4 Monet paintings, their photo version

Then run evaluate_local.py to write submission.csv and full_metrics_report.csv.

Usage (from the repo root):
  python task3_gan/sneha_singh/make_submission.py
  python task3_gan/sneha_singh/evaluate_local.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MEMBER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("score_checkpoints", MEMBER / "score_checkpoints.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

CKPT = MEMBER / "checkpoints" / "best.pt"
if not CKPT.exists():
    CKPT = MEMBER / "checkpoints" / "best_generators.pt"   # generator-only copy kept in git
SINGLE = [(False, 0, 0)]
SIX_VIEWS = [(False, 0, 0), (True, 0, 0), (False, 4, 4), (True, -4, -4), (False, -4, 4), (True, 4, -4)]


def shift(x, dx, dy):
    p = max(abs(dx), abs(dy), 1)
    return F.pad(x, (p, p, p, p), mode="reflect")[..., p + dy:p + dy + x.size(2), p + dx:p + dx + x.size(3)]


@torch.no_grad()
def translate(G, x, views):
    acc = 0
    for flip, dx, dy in views:
        y = shift(G(shift(x.flip(-1) if flip else x, dx, dy)), -dx, -dy)
        acc = acc + (y.flip(-1) if flip else y)
    return acc / len(views)


def to_pil(y):
    y = y[0].float().cpu().clamp(-1, 1)
    return Image.fromarray((((y + 1) * 0.5).permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def write_folder(G, paths, out_dir, views, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()
    for i, p in enumerate(paths):
        x = sc.TF(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        to_pil(translate(G, x, views)).save(out_dir / f"{i:05d}.jpg", quality=95)
    print("wrote", len(paths), "images to", out_dir, flush=True)


def write_grid(G_AB, G_BA, photos, monets, device, n=4, size=256):
    grid = Image.new("RGB", (4 * size, n * size), "white")
    for r in range(n):
        p = sc.TF(Image.open(photos[r]).convert("RGB")).unsqueeze(0).to(device)
        m = sc.TF(Image.open(monets[r]).convert("RGB")).unsqueeze(0).to(device)
        row = [to_pil(p), to_pil(translate(G_AB, p, SINGLE)), to_pil(m), to_pil(translate(G_BA, m, SIX_VIEWS))]
        for c, im in enumerate(row):
            grid.paste(im, (c * size, r * size))
    out = MEMBER / "outputs" / "samples" / "grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print("wrote", out, "(columns: photo, -> Monet, Monet, -> photo)", flush=True)


def main():
    device = sc.pick_device()
    state = torch.load(CKPT, map_location=device, weights_only=False)
    ngf = int(state.get("ngf", 64))
    G_AB = sc.ResnetGenerator("nearest", ngf=ngf).to(device).eval()  # photo -> Monet
    G_BA = sc.ResnetGenerator("nearest", ngf=ngf).to(device).eval()  # Monet -> photo
    G_AB.load_state_dict(state["G_AB"])
    G_BA.load_state_dict(state["G_BA"])
    print("checkpoint:", CKPT.name, "epoch", state.get("epoch"), "| ngf", ngf, "| device:", device, flush=True)

    monets = sc.ev._list_images(sc.DATA / "monet_jpg")
    photos = sc.ev._list_images(sc.DATA / "photo_jpg")
    write_grid(G_AB, G_BA, photos, monets, device)
    write_folder(G_BA, monets, MEMBER / "outputs" / "pred_A2B", SIX_VIEWS, device)
    write_folder(G_AB, photos, MEMBER / "outputs" / "pred_B2A", SINGLE, device)
    print("next: python task3_gan/sneha_singh/evaluate_local.py", flush=True)


if __name__ == "__main__":
    main()
