"""
Task 3.2(1) - visual check: a grid of real inputs next to their translations, both directions.

Columns: photo | photo->Monet (pred_B2A) | Monet | Monet->photo (pred_A2B).
Rows are fixed (evenly spaced through the sorted file lists), so the grid is comparable across runs.

Usage (after generate.py):
    python sample_grid.py            # writes ../outputs/sample_grid.png
"""
import argparse
import os

from PIL import Image

from dataset import list_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir_a", default="../../data/monet_jpg")
    ap.add_argument("--data_dir_b", default="../../data/photo_jpg")
    ap.add_argument("--out_dir", default="../outputs")
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    monets, photos = list_images(args.data_dir_a), list_images(args.data_dir_b)
    s, n = args.size, args.rows
    grid = Image.new("RGB", (4 * s, n * s), "white")
    for r in range(n):
        p = photos[r * len(photos) // n]
        m = monets[r * len(monets) // n]
        fake_m = os.path.join(args.out_dir, "pred_B2A", os.path.splitext(os.path.basename(p))[0] + ".jpg")
        fake_p = os.path.join(args.out_dir, "pred_A2B", os.path.splitext(os.path.basename(m))[0] + ".jpg")
        for c, path in enumerate([p, fake_m, m, fake_p]):
            grid.paste(Image.open(path).convert("RGB").resize((s, s)), (c * s, r * s))
    out = os.path.join(args.out_dir, "sample_grid.png")
    grid.save(out)
    print(f"Wrote {out}  (columns: photo | photo->Monet | Monet | Monet->photo)")


if __name__ == "__main__":
    main()
