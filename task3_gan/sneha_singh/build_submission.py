"""Build the submission outputs from a trained checkpoint, then score them with the TA's notebook.

- pred_B2A: all 7,038 photos -> Monet with G_AB (single pass)
- pred_A2B: all 300 Monets -> photo with G_BA (average of 6 flipped / shifted views, as in Sneha's submission)
Then runs Part3_Evaluation_Script.ipynb unchanged (via ta_score.py) and writes submission.csv next to the outputs.

Usage:
    python task3_gan/sneha_singh/build_submission.py --ckpt task3_gan/sneha_singh/checkpoints/best.pt --name final
"""
import argparse
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="best.pt or best_combo.pt (keys G_AB, G_BA)")
ap.add_argument("--name", required=True, help="output folder name under --out")
ap.add_argument("--ngf", type=int, default=None, help="default: read from the checkpoint")
ap.add_argument("--n_blocks", type=int, default=None, help="default: read from the checkpoint, else 9")
ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))   # repo root
ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "outputs" / "builds"))
args = ap.parse_args()

M = Path(args.repo) / "task3_gan" / "sneha_singh"
sys.path.insert(0, str(M))
import make_submission as ms  # translate(), views and to_pil() used for the submitted outputs

sc = ms.sc
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st = torch.load(args.ckpt, map_location=dev, weights_only=True)
ngf = args.ngf or int(st.get("ngf", 64))
n_blocks = args.n_blocks or int(st.get("n_blocks", 9))
G_AB = sc.ResnetGenerator("nearest", ngf=ngf, n_blocks=n_blocks).to(dev).eval()  # photo -> Monet
G_BA = sc.ResnetGenerator("nearest", ngf=ngf, n_blocks=n_blocks).to(dev).eval()  # Monet -> photo
G_AB.load_state_dict(st["G_AB"]); G_BA.load_state_dict(st["G_BA"])
print(f"checkpoint {args.ckpt} | ngf {ngf} | blocks {n_blocks} | epoch {st.get('epoch')} {st.get('weights', '')} | device {dev}",
      flush=True)

OUT = Path(args.out) / args.name
monets = sc.ev._list_images(sc.DATA / "monet_jpg")
photos = sc.ev._list_images(sc.DATA / "photo_jpg")
assert len(monets) == 300 and len(photos) == 7038, (len(monets), len(photos))


@torch.no_grad()
def write(G, paths, out_dir, views):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()
    for i, p in enumerate(paths):
        x = sc.TF(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
        ms.to_pil(ms.translate(G, x, views)).save(out_dir / f"{i:05d}.jpg", quality=95)
    print("wrote", len(paths), "images to", out_dir, flush=True)


write(G_BA, monets, OUT / "pred_A2B", ms.SIX_VIEWS)  # Monet -> photo, 6-view average
write(G_AB, photos, OUT / "pred_B2A", ms.SINGLE)     # photo -> Monet, single pass

# official score: the TA's notebook, unchanged except for the folder paths
ta = Path(__file__).resolve().parent / "ta_score.py"
subprocess.run([sys.executable, str(ta), "--gen_a2b", str(OUT / "pred_A2B"), "--gen_b2a", str(OUT / "pred_B2A")],
               cwd=OUT, check=True)
print("submission.csv written to", OUT / "submission.csv", flush=True)
