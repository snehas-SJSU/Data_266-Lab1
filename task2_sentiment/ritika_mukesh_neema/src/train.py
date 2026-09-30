"""
Task 2.2 - Model Training and Evaluation.

Trains all 3 models (baseline, bilstm, textcnn) on the preprocessed Yelp
Polarity split, times each one (wall-clock + examples/sec + peak memory),
evaluates each with the full Task 2 metrics list, runs the paired McNemar
test of baseline vs. each experimental model, and writes one combined
metrics_report.csv plus a per-model JSON with everything (including the
raw arrays error_analysis.py needs).
"""
import argparse
import json
import os
import resource
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import MODEL_REGISTRY
from metrics import full_report, mcnemar_test


class YelpDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = torch.from_numpy(X).long()
        self.y = torch.from_numpy(y).float()
        self.lengths = torch.from_numpy(lengths).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.lengths[i]


def train_one_model(name, model, train_loader, val_loader, device, epochs, lr):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    n_examples_total = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb, lb in train_loader:
            xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, lb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            n_examples_total += xb.size(0)
        print(f"  [{name}] epoch {epoch}: loss={epoch_loss/len(train_loader.dataset):.4f}")
    train_time = time.time() - t0
    examples_per_sec = n_examples_total / train_time

    peak_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "training_time_sec": train_time,
        "examples_per_sec": examples_per_sec,
        "peak_memory_mb": peak_mem_mb,
        "parameter_count": n_params,
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, preds, ys, lens = [], [], [], []
    for xb, yb, lb in loader:
        xb, lb = xb.to(device), lb.to(device)
        logits = model(xb, lb)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        preds.append((p >= 0.5).astype(int))
        ys.append(yb.numpy())
        lens.append(lb.cpu().numpy())
    return (np.concatenate(probs), np.concatenate(preds),
            np.concatenate(ys).astype(int), np.concatenate(lens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../data_processed")
    ap.add_argument("--out_dir", default="../outputs")
    ap.add_argument("--ckpt_dir", default="../checkpoints")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed_dim", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]

    def load_split(name):
        X = np.load(os.path.join(args.data_dir, f"X_{name}.npy"))
        y = np.load(os.path.join(args.data_dir, f"y_{name}.npy"))
        lengths = np.load(os.path.join(args.data_dir, f"lengths_{name}.npy"))
        return X, y, lengths

    Xtr, ytr, ltr = load_split("train")
    Xva, yva, lva = load_split("val")
    Xte, yte, lte = load_split("test")

    train_loader = DataLoader(YelpDataset(Xtr, ytr, ltr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(YelpDataset(Xva, yva, lva), batch_size=256, shuffle=False)
    test_loader = DataLoader(YelpDataset(Xte, yte, lte), batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_configs = {
        "baseline": dict(embed_dim=args.embed_dim),
        "bilstm": dict(embed_dim=args.embed_dim, hidden_dim=128),
        "textcnn": dict(embed_dim=args.embed_dim, num_filters=100, kernel_sizes=(3, 4, 5)),
    }

    all_reports = {}
    test_preds = {}   # for McNemar comparisons
    combined_rows = []

    for name, kwargs in model_configs.items():
        print(f"\n=== Training {name} ===")
        model = MODEL_REGISTRY[name](vocab_size=vocab_size, **kwargs)
        timing = train_one_model(name, model, train_loader, val_loader, device, args.epochs, args.lr)
        torch.save({"model_state": model.state_dict(), "config": kwargs, "meta": meta},
                   os.path.join(args.ckpt_dir, f"{name}.pt"))

        val_prob, val_pred, val_y, val_len = predict(model, val_loader, device)
        test_prob, test_pred, test_y, test_len = predict(model, test_loader, device)
        test_preds[name] = test_pred

        report = full_report(test_y, test_pred, test_prob, test_len, n_boot=args.n_boot)
        report.update(timing)
        report["device"] = str(device)
        all_reports[name] = report

        # keep raw arrays for error_analysis.py (not committed to the CSV report)
        np.savez(
            os.path.join(args.out_dir, f"{name}_test_predictions.npz"),
            y_true=test_y, y_pred=test_pred, y_prob=test_prob, lengths=test_len,
        )

        combined_rows.append({
            "model": name,
            "accuracy": report["accuracy"],
            "f1_macro": report["f1_macro"], "f1_micro": report["f1_micro"], "f1_weighted": report["f1_weighted"],
            "precision_macro": report["precision_macro"], "recall_macro": report["recall_macro"],
            "roc_auc": report["roc_auc"], "pr_auc": report["pr_auc"], "mcc": report["mcc"],
            "brier_score": report["brier_score"], "ece": report["ece"],
            "accuracy_ci": report["bootstrap_cis"]["accuracy_95ci"],
            "macro_f1_ci": report["bootstrap_cis"]["macro_f1_95ci"],
            "mcc_ci": report["bootstrap_cis"]["mcc_95ci"],
            "parameter_count": report["parameter_count"],
            "training_time_sec": report["training_time_sec"],
            "examples_per_sec": report["examples_per_sec"],
            "peak_memory_mb": report["peak_memory_mb"],
        })
        print(f"  test accuracy={report['accuracy']:.4f} macro-F1={report['f1_macro']:.4f} "
              f"ROC-AUC={report['roc_auc']:.4f}")

    # McNemar: baseline vs each experimental model
    mcnemar_results = {}
    for name in ["bilstm", "textcnn"]:
        mcnemar_results[f"baseline_vs_{name}"] = mcnemar_test(yte, test_preds["baseline"], test_preds[name])

    with open(os.path.join(args.out_dir, "full_metrics_report.json"), "w") as f:
        json.dump({"per_model": all_reports, "mcnemar_baseline_vs_experimental": mcnemar_results}, f, indent=2)

    pd.DataFrame(combined_rows).to_csv(os.path.join(args.out_dir, "metrics_report.csv"), index=False)

    with open(os.path.join(args.out_dir, "mcnemar_results.json"), "w") as f:
        json.dump(mcnemar_results, f, indent=2)

    print("\nDone. See outputs/metrics_report.csv, outputs/full_metrics_report.json, "
          "outputs/mcnemar_results.json, and outputs/<model>_test_predictions.npz")


if __name__ == "__main__":
    main()
