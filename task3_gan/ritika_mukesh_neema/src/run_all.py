"""
One-command pipeline for Task 3 (reproducibility requirement).

Usage (from GPU Lab, data already at task3_gan/data/monet_jpg and photo_jpg):
    cd task3_gan/ritika_neema/src
    python run_all.py

Smoke test (tiny run to sanity-check the whole pipeline in a couple minutes):
    python run_all.py --smoke_test
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
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--skip_lpips", action="store_true")
    args, unknown = ap.parse_known_args()

    if args.smoke_test:
        train_args = ["--n_epochs", "1", "--n_epochs_decay", "0", "--max_steps", "8",
                      "--n_blocks", "2", "--img_size", "64", "--save_every_epoch", "1"]
        eval_args = ["--n_blocks", "2", "--img_size", "64",
                     "--max_images_for_fid", "8", "--max_images_for_cycle", "4"]
        gen_args = ["--n_blocks", "2", "--img_size", "64"]
    else:
        train_args, gen_args, eval_args = [], [], []

    run([sys.executable, "train.py"] + train_args + unknown)
    run([sys.executable, "generate.py"] + gen_args)
    eval_cmd = [sys.executable, "evaluate_local.py"] + eval_args
    if args.skip_lpips:
        eval_cmd.append("--skip_lpips")
    run(eval_cmd)
    print("\nAll done. See ../outputs, ../checkpoints, ../full_metrics_report.csv, ../submission.csv")
    print("Next: python human_audit.py prepare   (then send outputs/human_audit/ to your 2 raters)")


if __name__ == "__main__":
    main()
