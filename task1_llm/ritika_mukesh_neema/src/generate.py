"""
Task 1.3(5) + 1.4 - Text generation and failure-case scaffolding.

Generates samples with greedy decoding and with temperature sampling,
computes generation-diversity metrics (distinct-1/2/3, repeated 4-gram rate)
and generation tokens/sec, and writes everything needed to do the failure
analysis by hand (the actual write-up of 3 failure cases belongs in
failure_analysis.md, informed by generated_samples.txt produced here).
"""
import argparse
import json
import os
import time

import torch

from model import GPTScratch
from metrics import distinct_n, repeated_4gram_rate


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]
    cfg = ckpt["config"]
    model = GPTScratch(
        vocab_size=meta["vocab_size"], block_size=meta["block_size"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"], dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../checkpoints/ckpt_final.pt")
    ap.add_argument("--data_dir", default="../data_processed")
    ap.add_argument("--out_dir", default="../outputs")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--gen_len", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--prompt", default="Once upon a time")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.ckpt, device)

    with open(os.path.join(args.data_dir, "char_to_idx.json")) as f:
        char_to_idx = json.load(f)
    with open(os.path.join(args.data_dir, "idx_to_char.json")) as f:
        idx_to_char = {int(k): v for k, v in json.load(f).items()}

    def encode(s):
        return [char_to_idx.get(c, 0) for c in s]

    def decode(ids):
        return "".join(idx_to_char[i] for i in ids)

    prompt_ids = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)

    results = {"greedy": [], "temperature_sampling": []}

    # --- greedy decoding ---
    t0 = time.time()
    for _ in range(args.n_samples // 2 or 1):
        out = model.generate(prompt_ids.clone(), max_new_tokens=args.gen_len, greedy=True)
        results["greedy"].append(decode(out[0].tolist()))
    greedy_time = time.time() - t0

    # --- temperature sampling ---
    t0 = time.time()
    for _ in range(args.n_samples // 2 or 1):
        out = model.generate(
            prompt_ids.clone(), max_new_tokens=args.gen_len,
            temperature=args.temperature, greedy=False, top_k=args.top_k,
        )
        results["temperature_sampling"].append(decode(out[0].tolist()))
    sample_time = time.time() - t0

    all_texts = results["greedy"] + results["temperature_sampling"]
    n_gen_tokens = len(all_texts) * args.gen_len
    total_gen_time = greedy_time + sample_time

    gen_metrics = {
        "distinct_1": distinct_n(all_texts, 1),
        "distinct_2": distinct_n(all_texts, 2),
        "distinct_3": distinct_n(all_texts, 3),
        "repeated_4gram_rate": repeated_4gram_rate(all_texts),
        "generation_tokens_per_sec": n_gen_tokens / total_gen_time,
        "n_samples_generated": len(all_texts),
        "gen_len_chars": args.gen_len,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "generated_samples.txt"), "w") as f:
        for mode, texts in results.items():
            f.write(f"\n===== {mode.upper()} =====\n")
            for i, t in enumerate(texts):
                f.write(f"\n--- sample {i} ---\n{t}\n")

    with open(os.path.join(args.out_dir, "generation_metrics.json"), "w") as f:
        json.dump(gen_metrics, f, indent=2)

    print(json.dumps(gen_metrics, indent=2))
    print(f"\nSamples written to {os.path.join(args.out_dir, 'generated_samples.txt')}")
    print("Use these to fill in failure_analysis.md with 3 real failure cases.")


if __name__ == "__main__":
    main()
