"""
One-command pipeline for Task 2 (reproducibility requirement).

Usage:
    cd task2_sentiment/ritika_neema/src
    python run_all.py --train_csv ../../data/yelp_train_40k.csv \
                       --test_csv  ../../data/yelp_test_8k.csv

Smoke test:
    python run_all.py --train_csv ... --test_csv ... --smoke_test
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
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--smoke_test", action="store_true")
    args, unknown = ap.parse_known_args()

    train_args = ["--epochs", "1", "--n_boot", "100"] if args.smoke_test else []

    run([sys.executable, "preprocess.py", "--train_csv", args.train_csv, "--test_csv", args.test_csv] + unknown)
    run([sys.executable, "train.py"] + train_args + unknown)
    run([sys.executable, "error_analysis.py", "--model_name", "textcnn"])
    print("\nAll done. See ../outputs and ../checkpoints.")


if __name__ == "__main__":
    main()
