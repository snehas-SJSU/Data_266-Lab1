"""
Task 3.1(4) + leaderboard submission prep.

Domain convention (see dataset.py / train.py): A = monet_jpg, B = photo_jpg.
  - pred_A2B/  = translations of monet paintings -> photo style   (G_A2B)
  - pred_B2A/  = translations of photos -> monet style            (G_B2A)
    This is the direction the Kaggle "I'm Something of a Painter Myself"
    competition actually scores: real photos translated into Monet style.

Output files keep their source filenames, so the TA's evaluation script (first
300 images of each folder, sorted by filename) sees the translations of the
first 300 photos / all 300 Monets. Images are written as JPEG quality 95.
Monet -> photo can average several flipped/shifted views (--views_a2b, see
inference.py); photo -> monet is always a single pass.

Writes pred_A2B/, pred_B2A/, a zip of pred_B2A, and generation_manifest.csv
(which checkpoint/config produced each file). The Kaggle upload itself is
submission.csv (ID,FID,MiFID), written by ../evaluate_local.py (the TA's script).
"""
import argparse
import csv
import os
import time
import zipfile

import torch
from torch.utils.data import DataLoader

from dataset import SingleDomainDataset
from inference import load_generators, translate, save_jpg


@torch.no_grad()
def translate_domain(generator, data_dir, out_dir, device, img_size, n_views=1, batch_size=8):
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):  # never mix in files from an older checkpoint
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(os.path.join(out_dir, f))
    ds = SingleDomainDataset(data_dir, img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    n_images = 0
    t0 = time.time()
    for batch in loader:
        fakes = translate(generator, batch["image"].to(device), n_views)
        for i, fname in enumerate(batch["filename"]):
            save_jpg(fakes[i], os.path.join(out_dir, os.path.splitext(fname)[0] + ".jpg"))
            n_images += 1
    return n_images, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../checkpoints/best.pt", help="best.pt, best_combo.pt or ckpt_final.pt")
    ap.add_argument("--data_dir_a", default="../../data/monet_jpg")
    ap.add_argument("--data_dir_b", default="../../data/photo_jpg")
    ap.add_argument("--out_dir", default="../outputs")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--views_a2b", type=int, default=None, choices=[1, 2, 6],
                    help="test-time views for monet->photo (default: the training config's value)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_A2B, G_B2A, cfg, ckpt = load_generators(args.ckpt, device)
    views_a2b = args.views_a2b or cfg.get("views_a2b", 1)
    print(f"Checkpoint {args.ckpt}: ngf={cfg.get('ngf', 64)} blocks={cfg.get('n_blocks', 9)} "
          f"upsample={cfg.get('upsample', 'convtranspose')} | epoch {ckpt.get('epoch', '-')} "
          f"{ckpt.get('weights', '')} {ckpt.get('selected', '')} | monet->photo views: {views_a2b}")

    pred_a2b_dir = os.path.join(args.out_dir, "pred_A2B")
    pred_b2a_dir = os.path.join(args.out_dir, "pred_B2A")

    n_a2b, t_a2b = translate_domain(G_A2B, args.data_dir_a, pred_a2b_dir, device, args.img_size, views_a2b)
    print(f"pred_A2B (monet->photo): {n_a2b} images in {t_a2b:.1f}s "
          f"({n_a2b/max(t_a2b,1e-8):.2f} img/s)")

    n_b2a, t_b2a = translate_domain(G_B2A, args.data_dir_b, pred_b2a_dir, device, args.img_size, 1)
    print(f"pred_B2A (photo->monet): {n_b2a} images in {t_b2a:.1f}s "
          f"({n_b2a/max(t_b2a,1e-8):.2f} img/s)  <-- Kaggle submission direction")

    zip_path = os.path.join(args.out_dir, "images.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(pred_b2a_dir)):
            zf.write(os.path.join(pred_b2a_dir, fname), arcname=fname)
    print(f"Wrote {zip_path}")

    manifest_path = os.path.join(os.path.dirname(os.path.abspath(args.out_dir)), "generation_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "direction", "checkpoint", "ngf", "n_blocks", "upsample", "views", "img_size"])
        for d, folder, views in [("monet_to_photo (A2B)", pred_a2b_dir, views_a2b), ("photo_to_monet (B2A)", pred_b2a_dir, 1)]:
            for fname in sorted(os.listdir(folder)):
                w.writerow([fname, d, args.ckpt, cfg.get("ngf", 64), cfg.get("n_blocks", 9),
                            cfg.get("upsample", "convtranspose"), views, args.img_size])
    print(f"Wrote {manifest_path}")
    print("Next: cd .. && python evaluate_local.py   (TA script: official FID / MiFID -> submission.csv)")


if __name__ == "__main__":
    main()
