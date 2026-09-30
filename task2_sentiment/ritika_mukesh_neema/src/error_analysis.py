"""
Task 2.2(4) - Manual error review: 20 of your own model's errors --
5 confident false positives, 5 confident false negatives, 5 near-threshold
errors, 5 slice-specific failures. Assigns an error type to each and drafts
a testable fix (edit the drafted fix text as needed after actually reading
the reviews -- the categorization/selection logic here is exact, the fix
proposals are a starting template for your own judgement call).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from metrics import length_slices


def select_confident_false_positives(y_true, y_pred, y_prob, k=5):
    mask = (y_pred == 1) & (y_true == 0)
    idx = np.where(mask)[0]
    idx = idx[np.argsort(-y_prob[idx])][:k]  # highest predicted-positive confidence
    return idx


def select_confident_false_negatives(y_true, y_pred, y_prob, k=5):
    mask = (y_pred == 0) & (y_true == 1)
    idx = np.where(mask)[0]
    idx = idx[np.argsort(y_prob[idx])][:k]  # lowest predicted-positive prob = most confident "negative"
    return idx


def select_near_threshold_errors(y_true, y_pred, y_prob, k=5):
    mask = y_pred != y_true
    idx = np.where(mask)[0]
    idx = idx[np.argsort(np.abs(y_prob[idx] - 0.5))][:k]
    return idx


def select_slice_specific_failures(y_true, y_pred, y_prob, lengths, k=5):
    slices = length_slices(lengths)
    err_rate_by_slice = {}
    for s in set(slices):
        m = slices == s
        if m.sum() > 0:
            err_rate_by_slice[s] = float((y_true[m] != y_pred[m]).mean())
    worst_slice = max(err_rate_by_slice, key=err_rate_by_slice.get)
    mask = (slices == worst_slice) & (y_pred != y_true)
    idx = np.where(mask)[0][:k]
    return idx, worst_slice


ERROR_TYPE_HINTS = {
    "confident_false_positive": "sarcasm / negation not captured; strong positive surface words despite a negative review",
    "confident_false_negative": "negative surface words used inside an overall positive review (e.g. 'not bad at all', comparative complaints followed by praise)",
    "near_threshold": "genuinely mixed / ambivalent review, or short review with too little signal",
    "slice_specific": "systematic weakness on this length bucket -- e.g. very short reviews give the model too few tokens to disambiguate",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="textcnn", help="which model's predictions to review")
    ap.add_argument("--data_dir", default="../data_processed")
    ap.add_argument("--out_dir", default="../outputs")
    args = ap.parse_args()

    pred_file = os.path.join(args.out_dir, f"{args.model_name}_test_predictions.npz")
    npz = np.load(pred_file)
    y_true, y_pred, y_prob, lengths = npz["y_true"], npz["y_pred"], npz["y_prob"], npz["lengths"]

    test_df = pd.read_csv(os.path.join(args.data_dir, "cleaned_test.csv"))

    fp_idx = select_confident_false_positives(y_true, y_pred, y_prob)
    fn_idx = select_confident_false_negatives(y_true, y_pred, y_prob)
    nt_idx = select_near_threshold_errors(y_true, y_pred, y_prob)
    slice_idx, worst_slice = select_slice_specific_failures(y_true, y_pred, y_prob, lengths)

    def rows(idx_list, err_type):
        out = []
        for i in idx_list:
            out.append({
                "error_type": err_type,
                "index": int(i),
                "true_label": int(y_true[i]),
                "pred_label": int(y_pred[i]),
                "pred_prob_positive": float(y_prob[i]),
                "review_length_tokens": int(lengths[i]),
                "text": test_df.iloc[i]["text"][:500],
                "suggested_hint": ERROR_TYPE_HINTS[err_type],
            })
        return out

    all_rows = (
        rows(fp_idx, "confident_false_positive")
        + rows(fn_idx, "confident_false_negative")
        + rows(nt_idx, "near_threshold")
        + rows(slice_idx, "slice_specific")
    )

    out_path = os.path.join(args.out_dir, f"error_analysis_{args.model_name}.json")
    with open(out_path, "w") as f:
        json.dump({"worst_slice": worst_slice, "cases": all_rows}, f, indent=2)

    # also emit a markdown skeleton ready to hand-annotate with a proposed fix per case
    md_lines = [f"# Task 2.2(4) — Manual Error Review ({args.model_name})\n",
                f"Worst-performing length slice on this model: **{worst_slice}**\n"]
    for i, row in enumerate(all_rows, 1):
        md_lines.append(f"## Case {i} — {row['error_type']}")
        md_lines.append(f"- true={row['true_label']}, pred={row['pred_label']}, "
                         f"P(positive)={row['pred_prob_positive']:.3f}, length={row['review_length_tokens']} tokens")
        md_lines.append(f"- text: \"{row['text']}\"")
        md_lines.append(f"- likely cause: {row['suggested_hint']}")
        md_lines.append("- **proposed testable fix:** _(edit this after reading the review)_\n")

    md_path = os.path.join(os.path.dirname(args.out_dir) or ".", "failure_analysis.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Wrote {out_path} and {md_path}")
    print(f"Worst slice: {worst_slice}")


if __name__ == "__main__":
    main()
