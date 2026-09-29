"""
One-command pipeline for Task 1 (reproducibility requirement: single documented
command). No hard-coded personal paths -- everything is relative / argparse-driven.

Usage:
    cd task1_llm/ritika_neema/src
    python run_all.py --train_txt ../../data/tinystories_train_100k.txt \
                       --val_txt   ../../data/tinystories_val_10k.txt

For a quick smoke test (~a couple minutes on CPU) rather than the full run:
    python run_all.py --train_txt ... --val_txt ... --smoke_test
"""
import argparse
import subprocess
import sys


def run(cmd):
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_txt", required=True)
    ap.add_argument("--val_txt", required=True)
    ap.add_argument("--smoke_test", action="store_true",
                     help="tiny run (small split + few steps) to verify the pipeline end-to-end")
    args, unknown = ap.parse_known_args()

    if args.smoke_test:
        data_args = ["--n_train_seq", "2000", "--n_val_seq", "400", "--block_size", "128"]
        train_args = ["--epochs", "2", "--batch_size", "32", "--max_steps", "60",
                      "--n_layer", "2", "--n_head", "2", "--n_embd", "64"]
    else:
        data_args, train_args = [], []

    run([sys.executable, "data.py", "--train_txt", args.train_txt, "--val_txt", args.val_txt] + data_args + unknown)
    run([sys.executable, "train.py"] + train_args + unknown)
    run([sys.executable, "generate.py"])
    print("\nAll done. See ../outputs and ../checkpoints.")


if __name__ == "__main__":
    main()
