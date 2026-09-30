"""
Task 2.1 - Data Preprocessing for Yelp Polarity.

Handles all common raw formats:
  - HF parquet (fancyzhx/yelp_polarity: train-00000-of-00001.parquet etc.),
    columns "label","text", label in {0,1} (0=negative,1=positive)
  - HF-streamed CSV with header "label,text", label in {0,1}
  - Classic Zhang et al. Kaggle CSV, NO header, columns are label,text, label in {1,2}
    (1=negative, 2=positive)
Format is auto-detected from the file extension (.parquet vs .csv).
All are normalized to label in {0,1} with 0=negative, 1=positive.

Produces:
  - EDA plots (class distribution, review length distribution) -> outputs/
  - cleaned_train.csv / cleaned_val.csv / cleaned_test.csv (if test provided)
  - vocab.json (word -> id, built from train split only, min_freq threshold)
  - token id sequences saved as .npy (padded/truncated to max_len)
"""
import argparse
import json
import os
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

STEMMER = PorterStemmer()
STOPWORDS = set(ENGLISH_STOP_WORDS)
_WORD_RE = re.compile(r"[a-z']+")


def load_raw_csv(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        df = df[["label", "text"]]
    else:
        # Try with header first (HF-style: label,text)
        df = pd.read_csv(path)
        cols = [c.lower() for c in df.columns]
        if "label" in cols and "text" in cols:
            df.columns = cols
            df = df[["label", "text"]]
        else:
            # No usable header -> assume classic Zhang format: label,title,text or label,text (no header)
            df = pd.read_csv(path, header=None)
            if df.shape[1] >= 3:
                df = df.iloc[:, [0, df.shape[1] - 1]]
            df.columns = ["label", "text"]

    # normalize label encoding: {1,2} (Zhang) -> {0,1}; leave {0,1} (HF) as-is
    uniq = set(pd.unique(df["label"].dropna()))
    if uniq <= {1, 2}:
        df["label"] = df["label"].map({1: 0, 2: 1})
    df["label"] = df["label"].astype(int)
    return df


def handle_missing_and_malformed(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip().str.len() > 0]
    df = df[df["label"].isin([0, 1])]
    df = df.drop_duplicates(subset=["text"])
    df = df.reset_index(drop=True)
    print(f"  dropped {n0 - len(df)} missing/malformed/duplicate rows ({n0} -> {len(df)})")
    return df


def clean_text(text: str, remove_stopwords=True, stem=True) -> str:
    text = str(text).lower()
    text = re.sub(r"<br\s*/?>", " ", text)          # stray HTML
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # urls
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = _WORD_RE.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        # Porter stemmer: pure algorithmic, no downloaded corpus needed
        # (chosen over WordNet lemmatization specifically for offline
        # reproducibility -- see README / results.md).
        tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)


def eda(df: pd.DataFrame, out_dir: str, tag: str):
    class_counts = df["label"].value_counts().sort_index()
    lengths_words = df["text"].astype(str).str.split().map(len)
    lengths_chars = df["text"].astype(str).str.len()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["negative (0)", "positive (1)"], class_counts.reindex([0, 1]).fillna(0).values)
    axes[0].set_title(f"Class distribution ({tag})")
    axes[0].set_ylabel("count")

    axes[1].hist(lengths_words, bins=50)
    axes[1].set_title(f"Review length in words ({tag})")
    axes[1].set_xlabel("words per review")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"eda_{tag}.png"), dpi=150)
    plt.close()

    stats = {
        "n_rows": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "class_balance_ratio": float(class_counts.min() / class_counts.max()) if len(class_counts) == 2 else None,
        "length_words_mean": float(lengths_words.mean()),
        "length_words_median": float(lengths_words.median()),
        "length_words_p95": float(lengths_words.quantile(0.95)),
        "length_chars_mean": float(lengths_chars.mean()),
    }
    return stats


def build_vocab(texts, min_freq=2, max_vocab=30000):
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    most_common = [w for w, c in counter.most_common() if c >= min_freq][: max_vocab - 2]
    word2idx = {"<pad>": 0, "<unk>": 1}
    for w in most_common:
        word2idx[w] = len(word2idx)
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


def encode_texts(texts, word2idx, max_len):
    unk = word2idx["<unk>"]
    pad = word2idx["<pad>"]
    out = np.full((len(texts), max_len), pad, dtype=np.int64)
    lengths = np.zeros(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [word2idx.get(w, unk) for w in t.split()][:max_len]
        out[i, : len(ids)] = ids
        lengths[i] = len(ids)
    return out, lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", default="../data_processed")
    ap.add_argument("--val_frac", type=float, default=0.1, help="carved out of train for validation")
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--max_vocab", type=int, default=30000)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading...")
    train_df = load_raw_csv(args.train_csv)
    test_df = load_raw_csv(args.test_csv)

    print("Handling missing/malformed rows...")
    train_df = handle_missing_and_malformed(train_df)
    test_df = handle_missing_and_malformed(test_df)

    print("Running EDA (pre-cleaning, on raw text)...")
    train_stats = eda(train_df, args.out_dir, "train_raw")
    test_stats = eda(test_df, args.out_dir, "test_raw")
    with open(os.path.join(args.out_dir, "eda_stats.json"), "w") as f:
        json.dump({"train": train_stats, "test": test_stats}, f, indent=2)
    print("  train:", train_stats)
    print("  test:", test_stats)

    print("Cleaning text (lowercase, punctuation strip, stopword removal, Porter stemming)...")
    train_df["clean_text"] = train_df["text"].map(clean_text)
    test_df["clean_text"] = test_df["text"].map(clean_text)
    train_df = train_df[train_df["clean_text"].str.len() > 0].reset_index(drop=True)
    test_df = test_df[test_df["clean_text"].str.len() > 0].reset_index(drop=True)

    print("Carving out validation split from train...")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(train_df))
    n_val = int(len(train_df) * args.val_frac)
    val_df = train_df.iloc[idx[:n_val]].reset_index(drop=True)
    train_df = train_df.iloc[idx[n_val:]].reset_index(drop=True)

    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    print("Building vocab from train split only...")
    word2idx, idx2word = build_vocab(train_df["clean_text"], args.min_freq, args.max_vocab)
    print(f"  vocab size = {len(word2idx)}")
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(word2idx, f)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X, lengths = encode_texts(df["clean_text"], word2idx, args.max_len)
        np.save(os.path.join(args.out_dir, f"X_{name}.npy"), X)
        np.save(os.path.join(args.out_dir, f"lengths_{name}.npy"), lengths)
        np.save(os.path.join(args.out_dir, f"y_{name}.npy"), df["label"].values)
        df[["label", "text", "clean_text"]].to_csv(
            os.path.join(args.out_dir, f"cleaned_{name}.csv"), index=False
        )

    meta = {
        "vocab_size": len(word2idx),
        "max_len": args.max_len,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "min_freq": args.min_freq, "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Done.", meta)


if __name__ == "__main__":
    main()
