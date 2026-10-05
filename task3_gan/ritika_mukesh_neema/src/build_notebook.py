#!/usr/bin/env python3
"""
Assembles part3_cyclegan.ipynb from the real source files and the real logged
outputs of your completed run. Run this FROM INSIDE:
    Data_266-Lab1/task3_gan/ritika_mukesh_neema/src/

    cd task3_gan/ritika_mukesh_neema/src
    python build_notebook.py

It reads your actual .py files and actual outputs/*.json|.csv|.log|.png (no
fabricated numbers), and writes part3_cyclegan.ipynb next to this script.
"""
import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MEMBER = os.path.dirname(HERE)          # .../ritika_mukesh_neema
OUT = os.path.join(MEMBER, "outputs")


def read(path, default=""):
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        print(f"  [warn] missing: {path}")
        return default
    # Windows console redirection (PowerShell "> file.log") is often UTF-16.
    for enc in ("utf-8", "utf-16", "utf-16-le", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def code_cell(source, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": outputs or [],
        "source": source.splitlines(keepends=True),
    }


def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def stream_output(text):
    return [{"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}]


def image_output(png_path):
    try:
        with open(png_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return [{
            "output_type": "display_data",
            "data": {"image/png": b64, "text/plain": ["<Figure>"]},
            "metadata": {},
        }]
    except FileNotFoundError:
        print(f"  [warn] missing image: {png_path}")
        return []


def tail_lines(text, n=50):
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:])


def main():
    cells = []

    cells.append(md_cell(
        "# Task 3 — CycleGAN Photo <-> Monet Style Transfer (v2)\n"
        "**Author:** Ritika Mukesh Neema\n\n"
        "Real source from `src/` and the member root, followed by the real outputs of the v2 run: "
        "training on a Colab NVIDIA A100 (artifacts in `outputs/`: `train_log.jsonl`, `train_metrics.json`, "
        "`checkpoint_scores.csv`, `loss_curves.png`), then generation, the TA's evaluation script and the "
        "full metrics on the submitted checkpoint `checkpoints/best.pt` (epoch 110), run locally on an "
        "RTX 4060 Laptop GPU (console logs in `outputs/console_*.log`). Nothing below is re-executed or "
        "hand-edited. See `results.md` and `failure_analysis.md`."
    ))

    for path, title in [
        (os.path.join(HERE, "config.json"), "## Run settings (`src/config.json`)"),
        (os.path.join(HERE, "dataset.py"), "## Dataset (unpaired domains, epoch length)"),
        (os.path.join(HERE, "models.py"), "## Models — ResNet generators + PatchGAN discriminators"),
        (os.path.join(HERE, "augment.py"), "## DiffAugment for the discriminators"),
        (os.path.join(HERE, "utils.py"), "## Utilities (image pool, LR schedule, EMA, thermal guard)"),
        (os.path.join(HERE, "inference.py"), "## Inference helpers (checkpoint loading, view averaging, JPEG saving)"),
        (os.path.join(HERE, "ta_eval.py"), "## TA evaluation code used during training (checkpoint scoring)"),
        (os.path.join(HERE, "train.py"), "## Training loop (LSGAN + cycle-consistency + identity loss)"),
        (os.path.join(HERE, "generate.py"), "## Generation (pred_A2B / pred_B2A)"),
        (os.path.join(MEMBER, "evaluate_local.py"), "## TA evaluation script (`evaluate_local.py`, writes `submission.csv`)"),
        (os.path.join(HERE, "full_metrics.py"), "## All other metrics (KID, precision/recall, cycle L1, LPIPS, content cosine)"),
        (os.path.join(HERE, "sample_grid.py"), "## Visual check grid"),
        (os.path.join(HERE, "human_audit.py"), "## Human audit prep/scoring"),
    ]:
        src = read(path)
        if not src:
            continue
        cells.append(md_cell(title))
        cells.append(code_cell(src))

    # --- training (Colab A100) ---
    train_metrics = read(os.path.join(OUT, "train_metrics.json"))
    if train_metrics:
        tm = json.loads(train_metrics)
        tm.pop("loss_history", None)
        cells.append(md_cell("## Run: `python train.py` (125 epochs, Colab NVIDIA A100)\n"
                             "Final `outputs/train_metrics.json` (the per-epoch loss history is in the file):"))
        cells.append(code_cell(
            "import json\n"
            "m = json.load(open('../outputs/train_metrics.json')); m.pop('loss_history')\n"
            "print(json.dumps(m, indent=2))",
            outputs=stream_output(json.dumps(tm, indent=2)),
        ))
    cells.append(md_cell("### Loss curves"))
    cells.append(code_cell(
        "from IPython.display import Image\nImage('../outputs/loss_curves.png')",
        outputs=image_output(os.path.join(OUT, "loss_curves.png")),
    ))
    scores = read(os.path.join(OUT, "checkpoint_scores.csv"))
    if scores:
        cells.append(md_cell("### Checkpoint scores (TA method, last 30 epochs, raw and EMA weights)"))
        cells.append(code_cell("print(open('../outputs/checkpoint_scores.csv').read())", outputs=stream_output(scores)))

    # --- generation, TA score, metrics (local) ---
    for log, title in [
        ("console_generate.log", "## Run: `python generate.py --ckpt ../checkpoints/best.pt`"),
        ("console_evaluate_local.log", "## Run: `python evaluate_local.py` (TA script -> submission.csv)"),
        ("console_full_metrics.log", "## Run: `python full_metrics.py --ckpt ../checkpoints/best.pt`"),
    ]:
        text = read(os.path.join(OUT, log))
        cells.append(md_cell(title + "\nReal captured output:"))
        cells.append(code_cell(f"print(open('../outputs/{log}').read())",
                               outputs=stream_output(tail_lines(text, 80) if text else f"({log} not found)")))

    sub = read(os.path.join(MEMBER, "submission.csv"))
    if sub:
        cells.append(md_cell("### submission.csv (Kaggle upload; leaderboard = -(FID + MiFID) / 2)"))
        cells.append(code_cell("print(open('../submission.csv').read())", outputs=stream_output(sub)))
    cells.append(md_cell("## Visual check: photo | photo->Monet | Monet | Monet->photo"))
    cells.append(code_cell(
        "from IPython.display import Image\nImage('../outputs/sample_grid.png')",
        outputs=image_output(os.path.join(OUT, "sample_grid.png")),
    ))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = os.path.join(HERE, "part3_cyclegan.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
