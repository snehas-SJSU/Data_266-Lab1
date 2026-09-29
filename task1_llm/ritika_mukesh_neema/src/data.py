"""
Task 1.1 - Data Preprocessing for character-level GPT on TinyStories.

Expects a plain-text file where individual stories are separated by the
literal marker `<|endofstory|>` (this is exactly what the streaming download
snippet you were given produces). If your file doesn't use that marker,
pass --no_split_marker and the whole file will be treated as one long stream.

Produces:
  - char_to_idx.json / idx_to_char.json  (vocab, built ONLY from training text)
  - train_ids.npy, val_ids.npy           (encoded corpora, int16)
  - a fixed number of (input, target) fixed-length sequences for train/val,
    built with a configurable stride so we can hit an EXACT requested count
    (default: 100,000 train sequences / 10,000 val sequences) regardless of
    how much raw text was supplied.
"""
import argparse
import json
import os
import numpy as np


def read_corpus(path: str, split_marker: str = "<|endofstory|>") -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if split_marker and split_marker in text:
        # Normalize story boundaries to a single newline so the boundary
        # itself is still visible to the char-level model (it's a real
        # signal: "story ended here") but doesn't blow up the vocab.
        text = text.replace(split_marker, "\n")
    return text


def build_vocab(text: str):
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char


def encode(text: str, char_to_idx: dict, unk_idx: int) -> np.ndarray:
    # Chunked + vectorized: a 1.5B-char corpus makes a single whole-string
    # UTF-32 conversion memory-heavy (~6GB+ intermediate buffers) on Colab's
    # RAM. Process in chunks to keep peak memory bounded, and give visible
    # progress instead of a silent multi-minute wait.
    n = len(text)
    lookup = np.full(0x110000, unk_idx, dtype=np.int32)
    for ch, idx in char_to_idx.items():
        lookup[ord(ch)] = idx
    out = np.empty(n, dtype=np.int16)
    chunk_size = 50_000_000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        codepoints = np.frombuffer(text[start:end].encode("utf-32-le"), dtype=np.uint32)
        out[start:end] = lookup[codepoints].astype(np.int16)
        print(f"    encoded {end:,} / {n:,} chars ({100*end/n:.1f}%)", flush=True)
    return out


def make_fixed_length_sequences(ids: np.ndarray, block_size: int, n_sequences: int, seed: int = 6638):
    """
    Slice `ids` into exactly n_sequences (x, y) pairs of length block_size,
    where y is x shifted by one character (next-char prediction target).
    Uses a stride computed to spread windows across the whole corpus; falls
    back to overlapping windows (with random offsets) if the corpus is too
    short to give n_sequences non-overlapping windows.
    """
    rng = np.random.default_rng(seed)
    max_start = len(ids) - block_size - 1
    if max_start <= 0:
        raise ValueError(
            f"Corpus too short ({len(ids)} chars) for block_size={block_size}. "
            "Provide more text or reduce block_size."
        )

    non_overlap_capacity = max_start // block_size
    if non_overlap_capacity >= n_sequences:
        starts = np.arange(n_sequences) * block_size
    else:
        # Not enough text for fully non-overlapping windows -> sample
        # (with replacement) uniformly random start positions instead.
        starts = rng.integers(0, max_start, size=n_sequences)

    X = np.empty((n_sequences, block_size), dtype=np.int64)
    Y = np.empty((n_sequences, block_size), dtype=np.int64)
    for i, s in enumerate(starts):
        X[i] = ids[s : s + block_size]
        Y[i] = ids[s + 1 : s + 1 + block_size]
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_txt", required=True, help="path to TinyStories train text file")
    ap.add_argument("--val_txt", required=True, help="path to TinyStories validation text file")
    ap.add_argument("--out_dir", default="../data_processed")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--n_train_seq", type=int, default=100_000)
    ap.add_argument("--n_val_seq", type=int, default=10_000)
    ap.add_argument("--split_marker", default="<|endofstory|>")
    ap.add_argument("--seed", type=int, default=6638)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Reading corpora...")
    train_text = read_corpus(args.train_txt, args.split_marker)
    val_text = read_corpus(args.val_txt, args.split_marker)
    print(f"  train chars: {len(train_text):,}   val chars: {len(val_text):,}")

    print("Building char vocabulary from TRAIN text only...")
    char_to_idx, idx_to_char = build_vocab(train_text)
    vocab_size = len(char_to_idx)
    unk_idx = char_to_idx.get(" ", 0)  # fallback for chars unseen in train (e.g. rare unicode in val)
    print(f"  vocab_size = {vocab_size}")

    with open(os.path.join(args.out_dir, "char_to_idx.json"), "w") as f:
        json.dump(char_to_idx, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "idx_to_char.json"), "w") as f:
        json.dump(idx_to_char, f, ensure_ascii=False, indent=2)

    print("Encoding...")
    train_ids = encode(train_text, char_to_idx, unk_idx)
    val_ids = encode(val_text, char_to_idx, unk_idx)
    np.save(os.path.join(args.out_dir, "train_ids.npy"), train_ids)
    np.save(os.path.join(args.out_dir, "val_ids.npy"), val_ids)

    print(f"Building {args.n_train_seq:,} train / {args.n_val_seq:,} val fixed-length sequences "
          f"(block_size={args.block_size})...")
    Xtr, Ytr = make_fixed_length_sequences(train_ids, args.block_size, args.n_train_seq, args.seed)
    Xval, Yval = make_fixed_length_sequences(val_ids, args.block_size, args.n_val_seq, args.seed + 1)

    np.save(os.path.join(args.out_dir, "X_train.npy"), Xtr)
    np.save(os.path.join(args.out_dir, "Y_train.npy"), Ytr)
    np.save(os.path.join(args.out_dir, "X_val.npy"), Xval)
    np.save(os.path.join(args.out_dir, "Y_val.npy"), Yval)

    meta = {
        "vocab_size": vocab_size,
        "block_size": args.block_size,
        "n_train_seq": int(Xtr.shape[0]),
        "n_val_seq": int(Xval.shape[0]),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(meta)


if __name__ == "__main__":
    main()
