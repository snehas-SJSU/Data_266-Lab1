"""
Task 3.1(4) + leaderboard submission prep.

Domain convention (see dataset.py / train.py): A = monet_jpg, B = photo_jpg.
  - pred_A2B/  = translations of monet paintings -> photo style   (G_A2B)
  - pred_B2A/  = translations of photos -> monet style            (G_B2A)
    This is the direction the Kaggle "I'm Something of a Painter Myself"
    competition actually scores: real photos translated into Monet style.

Writes pred_A2B/, pred_B2A/, a Kaggle-ready images.zip of pred_B2A, and a
submission.csv manifest (filenames + which checkpoint/config produced them --
the actual Kaggle upload is the images.zip; submission.csv is the repo's
local record of what got submitted, per the assignment's repo structure).
"""
import argparse
import csv
import os
import time
import zipfile

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import ResnetGenerator
from dataset import SingleDomainDataset
from utils import denorm


def load_generators(ckpt_path, device, n_blocks):
    ckpt = torch.load(ckpt_path, map_location=device)
    G_A2B = ResnetGenerator(n_blocks=n_blocks).to(device)
    G_B2A = ResnetGenerator(n_blocks=n_blocks).to(device)
    G_A2B.load_state_dict(ckpt["G_A2B"])
    G_B2A.load_state_dict(ckpt["G_B2A"])
    G_A2B.eval()
    G_B2A.eval()
    return G_A2B, G_B2A, ckpt.get("config", {})


@torch.no_grad()
def translate_domain(generator, data_dir, out_dir, device, img_size, batch_size=8):
    os.makedirs(out_dir, exist_ok=True)
    ds = SingleDomainDataset(data_dir, img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    n_images = 0
    t0 = time.time()
    for batch in loader:
        imgs = batch["image"].to(device)
        fakes = denorm(generator(imgs)).cpu()
        for i, fname in enumerate(batch["filename"]):
            save_image(fakes[i], os.path.join(out_dir, fname))
            n_images += 1
    elapsed = time.time() - t0
    return n_images, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../checkpoints/ckpt_final.pt")
    ap.add_argument("--data_dir_a", default="../../data/monet_jpg")
    ap.add_argument("--data_dir_b", default="../../data/photo_jpg")
    ap.add_argument("--out_dir", default="../outputs")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--n_blocks", type=int, default=9)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_A2B, G_B2A, ckpt_config = load_generators(args.ckpt, device, args.n_blocks)

    pred_a2b_dir = os.path.join(args.out_dir, "pred_A2B")
    pred_b2a_dir = os.path.join(args.out_dir, "pred_B2A")

    n_a2b, t_a2b = translate_domain(G_A2B, args.data_dir_a, pred_a2b_dir, device, args.img_size)
    print(f"pred_A2B (monet->photo): {n_a2b} images in {t_a2b:.1f}s "
          f"({n_a2b/max(t_a2b,1e-8):.2f} img/s)")

    n_b2a, t_b2a = translate_domain(G_B2A, args.data_dir_b, pred_b2a_dir, device, args.img_size)
    print(f"pred_B2A (photo->monet): {n_b2a} images in {t_b2a:.1f}s "
          f"({n_b2a/max(t_b2a,1e-8):.2f} img/s)  <-- Kaggle submission direction")

    # zip pred_B2A for Kaggle upload
    zip_path = os.path.join(args.out_dir, "images.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(pred_b2a_dir)):
            zf.write(os.path.join(pred_b2a_dir, fname), arcname=fname)
    print(f"Wrote {zip_path} -- upload THIS file to the Kaggle competition.")

    # submission.csv manifest (repo record, not the Kaggle upload itself)
    manifest_path = os.path.join(os.path.dirname(args.out_dir) or ".", "submission.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "direction", "checkpoint", "n_blocks", "img_size"])
        for fname in sorted(os.listdir(pred_b2a_dir)):
            w.writerow([fname, "photo_to_monet (B2A)", args.ckpt, args.n_blocks, args.img_size])
    print(f"Wrote {manifest_path} (manifest of what's in images.zip)")


if __name__ == "__main__":
    main()
