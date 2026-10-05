"""
Official (TA) FID / MiFID, computed by the course's own evaluation notebook.

The function definitions are executed straight from task3_gan/Part3_Evaluation_Script.ipynb
(list_images, take_n, Inception-v3 feature extractor, frechet_distance, calculate_fid_mifid),
unchanged, so every score here is exactly what the TA's script computes:
first 300 images of each folder sorted by filename, FID on Inception pool features,
MiFID = mean cosine distance paired by index, submission = mean of both directions.

Used by train.py to score the last epochs while training. The final submission.csv is written by
../evaluate_local.py, which is the same TA notebook code as a script.
"""
import json
import os
import tempfile
from pathlib import Path

import torch
from PIL import Image

from dataset import make_transform
from inference import save_jpg, translate

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NOTEBOOK = os.path.join(HERE, "..", "..", "Part3_Evaluation_Script.ipynb")
N_EVAL = 300


def load_ta_functions(notebook=DEFAULT_NOTEBOOK, device=None):
    """Execute the notebook's import and function-definition cells; skip its config cell and the
    cells that run the evaluation or write submission.csv (those are re-done in official_scores)."""
    nb = json.load(open(notebook, encoding="utf-8"))
    ns = {"__name__": "ta_notebook",
          "device": device or torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        runs_eval = any(k in src for k in ("= calculate_fid_mifid(", "to_csv", "assert os.path.isdir", "BASE ="))
        if not runs_eval and (src.lstrip().startswith("import") or "def " in src):
            exec(compile(src, "Part3_Evaluation_Script.ipynb", "exec"), ns)
    return ns


def official_scores(gen_a2b_dir, gen_b2a_dir, real_monet_dir, real_photo_dir, ta):
    """The TA notebook's evaluation cells: both directions, then the submission averages."""
    real_monet = ta["take_n"](ta["list_images"](str(real_monet_dir)), N_EVAL)
    real_photo = ta["take_n"](ta["list_images"](str(real_photo_dir)), N_EVAL)
    gen_a2b = ta["take_n"](ta["list_images"](str(gen_a2b_dir)), N_EVAL)
    gen_b2a = ta["take_n"](ta["list_images"](str(gen_b2a_dir)), N_EVAL)
    fid_b2a, mifid_b2a = ta["calculate_fid_mifid"](real_monet, gen_b2a)  # photo -> monet
    fid_a2b, mifid_a2b = ta["calculate_fid_mifid"](real_photo, gen_a2b)  # monet -> photo
    fid, mifid = (fid_a2b + fid_b2a) / 2, (mifid_a2b + mifid_b2a) / 2
    return {"fid_b2a": fid_b2a, "mifid_b2a": mifid_b2a, "fid_a2b": fid_a2b, "mifid_a2b": mifid_a2b,
            "fid": fid, "mifid": mifid, "score": (fid + mifid) / 2}


@torch.no_grad()
def _write_translations(G, paths, out_dir, device, n_views, img_size):
    tf = make_transform(img_size, train=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    G.eval()
    for p in paths:  # keep source filenames so the TA's sorted-first-300 picks the same inputs
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        save_jpg(translate(G, x, n_views)[0], out_dir / os.path.basename(p))


def score_generators(G_A2B, G_B2A, monet_files, photo_files, monet_dir, photo_dir, ta, device,
                     views_a2b=6, img_size=256, n_eval=N_EVAL):
    """Translate the images the TA script will look at (first n_eval sorted of each domain)
    and score them with the TA's code. Returns the same dict as official_scores()."""
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        _write_translations(G_A2B, sorted(monet_files)[:n_eval], t / "pred_A2B", device, views_a2b, img_size)
        _write_translations(G_B2A, sorted(photo_files)[:n_eval], t / "pred_B2A", device, 1, img_size)
        return official_scores(t / "pred_A2B", t / "pred_B2A", monet_dir, photo_dir, ta)
