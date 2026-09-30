"""
Task 2 - Evaluation Metrics (report all, per model).

Implements every metric in the assignment's Task 2 metrics list:
  accuracy; precision/recall/F1 (macro, micro, weighted); confusion matrix;
  ROC-AUC; PR-AUC; MCC; Brier score; expected calibration error (ECE);
  95% bootstrap CIs for accuracy/macro-F1/MCC; paired McNemar test
  (baseline vs each experimental model); macro-F1 and error rate per slice;
  parameter count / training time / examples-per-sec / peak memory (these
  three are timed in train.py and merged in here).
"""
import numpy as np
from scipy.stats import chi2, binomtest
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score, matthews_corrcoef, brier_score_loss,
)


def basic_classification_metrics(y_true, y_pred, y_prob):
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ["macro", "micro", "weighted"]:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out[f"precision_{avg}"] = float(p)
        out[f"recall_{avg}"] = float(r)
        out[f"f1_{avg}"] = float(f1)
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    out["brier_score"] = float(brier_score_loss(y_true, y_prob))
    return out


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(y_prob)) * abs(acc - conf)
    return float(ece)


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = metric_fn(y_true[idx], y_pred[idx])
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return {"point": float(metric_fn(y_true, y_pred)), "ci_lower": lo, "ci_upper": hi}


def macro_f1_metric(y_true, y_pred):
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return f1


def bootstrap_all_cis(y_true, y_pred, n_boot=1000, seed=42):
    return {
        "accuracy_95ci": bootstrap_ci(y_true, y_pred, accuracy_score, n_boot, seed=seed),
        "macro_f1_95ci": bootstrap_ci(y_true, y_pred, macro_f1_metric, n_boot, seed=seed),
        "mcc_95ci": bootstrap_ci(y_true, y_pred, matthews_corrcoef, n_boot, seed=seed),
    }


def mcnemar_test(y_true, pred_a, pred_b):
    """
    Paired McNemar test comparing two models' correctness on the SAME
    validation/test examples. Uses the exact binomial test when the
    discordant-pair count is small (<25), otherwise the standard
    continuity-corrected chi-square approximation -- avoids depending on
    statsmodels, which isn't part of this environment's manifest.
    """
    y_true = np.asarray(y_true)
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))   # a wrong, b right
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "degenerate_no_discordant_pairs"}
    if n < 25:
        res = binomtest(min(b, c), n, 0.5)
        return {"b": b, "c": c, "statistic": None, "p_value": float(res.pvalue), "method": "exact_binomial"}
    stat = (abs(b - c) - 1) ** 2 / n
    p = float(1 - chi2.cdf(stat, df=1))
    return {"b": b, "c": c, "statistic": float(stat), "p_value": p, "method": "chi2_continuity_corrected"}


def slice_metrics(y_true, y_pred, slice_labels):
    """
    Per-slice macro-F1 and error rate. `slice_labels` is an array-like of the
    same length as y_true/y_pred giving each example's slice name (e.g. a
    review-length bucket, defined by the caller).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    slice_labels = np.asarray(slice_labels)
    out = {}
    for s in sorted(set(slice_labels)):
        mask = slice_labels == s
        if mask.sum() == 0:
            continue
        _, _, f1, _ = precision_recall_fscore_support(
            y_true[mask], y_pred[mask], average="macro", zero_division=0
        )
        err_rate = float((y_true[mask] != y_pred[mask]).mean())
        out[str(s)] = {"n": int(mask.sum()), "macro_f1": float(f1), "error_rate": err_rate}
    return out


def length_slices(lengths, edges=(0, 20, 60, 10_000), names=("short", "medium", "long")):
    lengths = np.asarray(lengths)
    labels = np.array(["unassigned"] * len(lengths), dtype=object)
    for i, name in enumerate(names):
        mask = (lengths >= edges[i]) & (lengths < edges[i + 1])
        labels[mask] = name
    return labels


def full_report(y_true, y_pred, y_prob, lengths, n_boot=1000):
    report = basic_classification_metrics(y_true, y_pred, y_prob)
    report["ece"] = expected_calibration_error(y_true, y_prob)
    report["bootstrap_cis"] = bootstrap_all_cis(y_true, y_pred, n_boot=n_boot)
    slices = length_slices(lengths)
    report["slice_metrics_by_length"] = slice_metrics(y_true, y_pred, slices)
    return report
