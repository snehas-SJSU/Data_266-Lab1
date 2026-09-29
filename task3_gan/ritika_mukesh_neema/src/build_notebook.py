#!/usr/bin/env python3
"""
Assembles part3_cyclegan.ipynb from the real source files and the real logged
outputs of your completed run. Run this FROM INSIDE:
    Data_266-Lab1/task3_gan/ritika_mukesh_neema/src/

    cd ~/Desktop/Data_266-Lab1/task3_gan/ritika_mukesh_neema/src
    python3 build_notebook.py

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
            "data": {"image/png": b64, "text/plain": ["<Figure: loss curves>"]},
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
        "# Task 3 — CycleGAN Photo <-> Monet Style Transfer\n"
        "**Author:** Ritika Mukesh Neema\n\n"
        "Real source from `src/*.py` (ResNet generators, PatchGAN discriminators, "
        "LSGAN + cycle-consistency + identity loss, image replay buffer), followed "
        "by the real captured console output and metrics from the completed "
        "80-epoch run on an RTX 5090 (see `results.md`, `outputs/console_train.log`, "
        "`outputs/full_metrics_report.csv`)."
    ))

    for fname, title in [
        ("dataset.py", "## Dataset"),
        ("utils.py", "## Utilities (image buffer, checkpointing, etc.)"),
        ("models.py", "## Models — ResNet generators + PatchGAN discriminators"),
        ("train.py", "## Training loop (LSGAN + cycle-consistency + identity loss)"),
        ("generate.py", "## Generation (translate held-out photos/Monets)"),
        ("evaluate_local.py", "## Local evaluation (FID / KID / LPIPS / cycle-L1 / content-cosine)"),
        ("human_audit.py", "## Human audit prep/scoring"),
    ]:
        path = os.path.join(HERE, fname)
        src = read(path)
        if not src:
            continue
        cells.append(md_cell(title))
        cells.append(code_cell(src))

    # --- real training output ---
    console_train = read(os.path.join(OUT, "console_train.log"))
    train_metrics = read(os.path.join(OUT, "train_metrics.json"))

    cells.append(md_cell("## Run: `python train.py` (80 epochs, RTX 5090)\n"
                          "Real console output from the completed training run:"))
    cells.append(code_cell(
        "# Captured from outputs/console_train.log (last 50 lines) — not re-executed here\n"
        "print(open('../outputs/console_train.log').read())",
        outputs=stream_output(tail_lines(console_train, 50) if console_train else "(console_train.log not found)"),
    ))
    if train_metrics:
        cells.append(md_cell("### Final train_metrics.json"))
        cells.append(code_cell(
            "import json\nprint(json.dumps(json.load(open('../outputs/train_metrics.json')), indent=2))",
            outputs=stream_output(json.dumps(json.loads(train_metrics), indent=2)),
        ))

    cells.append(md_cell("### Loss curves"))
    cells.append(code_cell(
        "from IPython.display import Image\nImage('../outputs/loss_curves.png')",
        outputs=image_output(os.path.join(OUT, "loss_curves.png")),
    ))

    # --- generation output ---
    console_generate = read(os.path.join(OUT, "console_generate.log"))
    cells.append(md_cell("## Run: `python generate.py`\nReal captured output:"))
    cells.append(code_cell(
        "print(open('../outputs/console_generate.log').read())",
        outputs=stream_output(console_generate if console_generate else "(console_generate.log not found)"),
    ))

    # --- eval output ---
    console_eval = read(os.path.join(OUT, "console_eval.log"))
    full_metrics_csv = read(os.path.join(OUT, "full_metrics_report.csv"))
    full_metrics_json = read(os.path.join(OUT, "full_metrics_report.json"))

    cells.append(md_cell("## Run: `python evaluate_local.py`\nReal captured output:"))
    cells.append(code_cell(
        "print(open('../outputs/console_eval.log').read())",
        outputs=stream_output(tail_lines(console_eval, 60) if console_eval else "(console_eval.log not found)"),
    ))
    if full_metrics_json:
        cells.append(md_cell("### full_metrics_report.json"))
        cells.append(code_cell(
            "import json\nprint(json.dumps(json.load(open('../outputs/full_metrics_report.json')), indent=2))",
            outputs=stream_output(json.dumps(json.loads(full_metrics_json), indent=2)),
        ))
    if full_metrics_csv:
        cells.append(md_cell("### full_metrics_report.csv"))
        cells.append(code_cell(
            "print(open('../outputs/full_metrics_report.csv').read())",
            outputs=stream_output(full_metrics_csv),
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
