"""
One-command pipeline for Task 3 (reproducibility requirement).

Usage (data already at task3_gan/data/monet_jpg and photo_jpg):
    cd task3_gan/ritika_mukesh_neema/src
    python run_all.py

Steps: train.py (settings from config.json; resumes automatically from
../checkpoints/last.pt) -> generate.py (best checkpoint) -> ../evaluate_local.py (the TA's
evaluation script -> submission.csv) -> full_metrics.py (all other metrics -> full_metrics_report.csv)
-> sample_grid.py -> metrics_report.py (required metrics in one file -> metrics_report.csv).

Smoke test (tiny run in separate folders, a few minutes, never touches the real run):
    python run_all.py --smoke_test
Extra flags are passed to train.py, e.g.  python run_all.py --ckpt_dir /content/drive/MyDrive/ritika_ckpts
"""
import argparse
import subprocess
import sys


def run(cmd):
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--skip_lpips", action="store_true")
    ap.add_argument("--ckpt", default=None, help="checkpoint for generate.py (default: <ckpt_dir>/best.pt)")
    args, train_extra = ap.parse_known_args()

    ckpt_dir, out_dir, sub = "../checkpoints", "../outputs", "../submission.csv"
    if "--ckpt_dir" in train_extra:
        ckpt_dir = train_extra[train_extra.index("--ckpt_dir") + 1]
    if args.smoke_test:
        ckpt_dir, out_dir, sub = "../checkpoints_smoke", "../outputs_smoke", "../outputs_smoke/submission_smoke.csv"
        train_extra = ["--n_epochs", "1", "--n_epochs_decay", "1", "--steps_per_epoch", "4", "--n_blocks", "2",
                       "--score_last", "1", "--resume_from", "", "--ckpt_dir", ckpt_dir, "--out_dir", out_dir] + train_extra

    run([sys.executable, "train.py"] + train_extra)
    run([sys.executable, "generate.py", "--ckpt", args.ckpt or f"{ckpt_dir}/best.pt", "--out_dir", out_dir])
    run([sys.executable, "../evaluate_local.py", out_dir, sub])
    eval_cmd = [sys.executable, "full_metrics.py", "--ckpt", args.ckpt or f"{ckpt_dir}/best.pt", "--out_dir", out_dir]
    if args.smoke_test:
        eval_cmd += ["--max_images_for_fid", "16", "--max_images_for_cycle", "4"]
    if args.skip_lpips:
        eval_cmd.append("--skip_lpips")
    run(eval_cmd)
    if not args.smoke_test:
        run([sys.executable, "sample_grid.py"])
        run([sys.executable, "metrics_report.py"])  # required metrics in one file -> ../metrics_report.csv
    print(f"\nAll done. Outputs in {out_dir}, checkpoints in {ckpt_dir}, Kaggle file {sub}")
    print("Next: python human_audit.py prepare   (then send outputs/human_audit/ to your 2 raters)")


if __name__ == "__main__":
    main()
