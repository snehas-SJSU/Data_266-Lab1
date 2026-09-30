#!/usr/bin/env python3
"""
Assembles part2_sentiment.ipynb from the real source files and the real
logged outputs of your completed run. Run this FROM INSIDE:
    Data_266-Lab1/task2_sentiment/ritika_mukesh_neema/src/

    cd ~/Desktop/Data_266-Lab1/task2_sentiment/ritika_mukesh_neema/src
    python3 build_notebook.py

It reads your actual .py files and actual outputs/*.csv|.json plus
console_run_all_full.log (placed next to this script beforehand), and
writes part2_sentiment.ipynb next to this script.
"""
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


def main():
    cells = []

    cells.append(md_cell(
        "# Task 2 — Yelp Polarity Sentiment Classification Bake-off\n"
        "**Author:** Ritika Mukesh Neema\n\n"
        "Three from-scratch-embedding models compared: a mean-pooled-embedding "
        "baseline, a BiLSTM, and a TextCNN (Kim 2014 style). Code cells below are "
        "the real source from `src/*.py`; output cells are the real captured "
        "console output and metrics from the completed full run on the full "
        "Yelp Polarity dataset (560,000 train / 38,000 test rows, downloaded from "
        "Hugging Face `fancyzhx/yelp_polarity`)."
    ))

    for fname, title in [
        ("preprocess.py", "## Preprocessing (cleaning, stemming, vocab, train/val/test split)"),
        ("models.py", "## Models — mean-pooled baseline, BiLSTM, TextCNN"),
        ("metrics.py", "## Metrics utilities (bootstrap CIs, calibration, McNemar test)"),
        ("train.py", "## Training loop (all 3 models)"),
        ("error_analysis.py", "## Error analysis"),
    ]:
        path = os.path.join(HERE, fname)
        src = read(path)
        if not src:
            continue
        cells.append(md_cell(title))
        cells.append(code_cell(src))

    console_log = read(os.path.join(HERE, "console_run_all_full.log"))
    cells.append(md_cell("## Run: `python run_all.py --train_csv ... --test_csv ...` (full run)\n"
                          "Real captured console output from the completed run:"))
    cells.append(code_cell(
        "# Captured console output from the completed full run — not re-executed here\n"
        "print(open('console_run_all_full.log').read())",
        outputs=stream_output(console_log if console_log else "(console_run_all_full.log not found)"),
    ))

    metrics_csv = read(os.path.join(OUT, "metrics_report.csv"))
    if metrics_csv:
        cells.append(md_cell("### outputs/metrics_report.csv (all 3 models, full metric suite)"))
        cells.append(code_cell(
            "print(open('../outputs/metrics_report.csv').read())",
            outputs=stream_output(metrics_csv),
        ))

    mcnemar = read(os.path.join(OUT, "mcnemar_results.json"))
    if mcnemar:
        cells.append(md_cell("### Paired McNemar test (baseline vs. each experimental model)"))
        cells.append(code_cell(
            "import json\nprint(json.dumps(json.load(open('../outputs/mcnemar_results.json')), indent=2))",
            outputs=stream_output(json.dumps(json.loads(mcnemar), indent=2)),
        ))

    err = read(os.path.join(OUT, "error_analysis_textcnn.json"))
    if err:
        cells.append(md_cell("### Error analysis (TextCNN) — worst slice + flagged cases\n"
                              "See `failure_analysis.md` for the hand-annotated write-up."))
        cells.append(code_cell(
            "import json\nd = json.load(open('../outputs/error_analysis_textcnn.json'))\n"
            "print('worst_slice:', d['worst_slice'])\nprint('n_cases:', len(d['cases']))",
            outputs=stream_output(
                "worst_slice: " + json.loads(err)["worst_slice"] + "\n"
                "n_cases: " + str(len(json.loads(err)["cases"]))
            ),
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

    out_path = os.path.join(HERE, "part2_sentiment.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
