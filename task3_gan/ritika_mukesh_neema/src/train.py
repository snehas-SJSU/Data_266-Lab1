"""
Task 3.1(2,3,5) - CycleGAN training: adversarial (LSGAN) + cycle-consistency
+ identity loss, image pool for discriminator stability, linear LR decay.
Logs everything needed for Task 3's "analyse training behaviour" requirement:
raw per-step JSONL log (do not hand-edit -- evidence trail), gen/disc loss
curves, cycle/identity loss curves, gradient norms + NaN/Inf count, images/sec,
peak memory, total training time, parameter counts.

v2 additions (all optional, switched on in config.json):
  - config.json holds the run's settings; command-line flags override it
  - steps_per_epoch: epoch length decoupled from the 300-image Monet set
  - resize-convolution upsampling in the generator (no checkerboard artifacts)
  - DiffAugment on everything the discriminators see (augment.py)
  - EMA copies of both generators (utils.EMA)
  - mixed precision (AMP) and a full auto-resume state (ckpt_dir/last.pt)
  - the last `score_last` epochs are scored with the TA's evaluation notebook
    (ta_eval.py); best.pt keeps the best epoch, and because photo->monet is scored
    only through G_B2A and monet->photo only through G_A2B, the best generator for
    each direction is also kept separately and paired in best_combo.pt
  - one-sided label smoothing for the discriminators (real target < 1)
  - optional GPU thermal guard for laptop runs
"""
import argparse
import csv
import json
import os
import random
import time

try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import ResnetGenerator, PatchGANDiscriminator, init_weights, num_params
from dataset import UnpairedImageDataset, list_images
from utils import ImagePool, lambda_lr, EMA, cool_down
from augment import diff_augment
import ta_eval

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    """Settings from config.json (keys starting with '_' and the free-text note are skipped)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        cfg = json.load(f)
    return {k: v for k, v in cfg.items() if not k.startswith("_") and k not in ("note", "smoke")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"),
                    help="JSON file with defaults for the options below; command-line flags override it")
    ap.add_argument("--data_dir_a", default="../../data/monet_jpg", help="domain A (e.g. Monet paintings)")
    ap.add_argument("--data_dir_b", default="../../data/photo_jpg", help="domain B (e.g. photos)")
    ap.add_argument("--out_dir", default="../outputs")
    ap.add_argument("--ckpt_dir", default="../checkpoints")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--n_epochs", type=int, default=10, help="epochs at full LR")
    ap.add_argument("--n_epochs_decay", type=int, default=10, help="epochs of linear LR decay to 0")
    ap.add_argument("--steps_per_epoch", type=int, default=0,
                    help="optimizer steps per epoch (0 = one pass over the smaller domain, the v1 behaviour)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--lambda_cycle", type=float, default=10.0)
    ap.add_argument("--lambda_identity", type=float, default=5.0)
    ap.add_argument("--ngf", type=int, default=64, help="generator base filters")
    ap.add_argument("--n_blocks", type=int, default=9, help="generator residual blocks (6 for 128px, 9 for 256px)")
    ap.add_argument("--upsample", choices=["convtranspose", "nearest"], default="convtranspose")
    ap.add_argument("--pool_size", type=int, default=50)
    ap.add_argument("--real_label", type=float, default=1.0,
                    help="target for real images in the discriminator loss (0.9 = one-sided label smoothing)")
    ap.add_argument("--grad_clip", type=float, default=10.0, help="max gradient norm for G and D (0 = no clipping)")
    ap.add_argument("--diffaug", default="", help='DiffAugment policy for the discriminators, e.g. "translation,cutout"')
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay for the generators (0 = off)")
    ap.add_argument("--amp", type=int, default=0, help="1 = mixed precision on CUDA")
    ap.add_argument("--score_last", type=int, default=0, help="score the last N epochs with the TA's notebook")
    ap.add_argument("--views_a2b", type=int, default=1, choices=[1, 2, 6], help="test-time views for monet->photo")
    ap.add_argument("--ta_notebook", default=ta_eval.DEFAULT_NOTEBOOK)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_temp", type=int, default=0, help="laptop thermal guard: pause at this GPU temperature (0 = off)")
    ap.add_argument("--resume_temp", type=int, default=74)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every_epoch", type=int, default=1, help="also keep ckpt_epoch{N}.pt every N epochs (0 = never)")
    ap.add_argument("--max_steps", type=int, default=None, help="optional hard cap for smoke tests")
    ap.add_argument("--resume_from", default="auto",
                    help='"auto" = continue from ckpt_dir/last.pt if it exists; a path = resume from that checkpoint; "" = fresh run')
    pre, _ = ap.parse_known_args()
    ap.set_defaults(**load_config(pre.config))
    args = ap.parse_args()
    if os.name == "nt":
        args.workers = 0  # Windows: DataLoader worker processes need a __main__ guard per worker; keep it simple

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    print(f"Device: {device} | AMP: {use_amp}")
    print("Config:", json.dumps(vars(args)))

    epoch_size = args.steps_per_epoch * args.batch_size if args.steps_per_epoch else None
    dataset = UnpairedImageDataset(args.data_dir_a, args.data_dir_b, args.img_size, train=True, epoch_size=epoch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        drop_last=True, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    print(f"Domain A (monet-style): {len(dataset.files_a)} images | Domain B (photo-style): {len(dataset.files_b)} images "
          f"| {len(loader)} steps/epoch")

    gen_kw = dict(ngf=args.ngf, n_blocks=args.n_blocks, upsample=args.upsample)
    G_A2B = init_weights(ResnetGenerator(**gen_kw)).to(device)  # monet -> photo
    G_B2A = init_weights(ResnetGenerator(**gen_kw)).to(device)  # photo -> monet (Kaggle-scored direction)
    D_A = init_weights(PatchGANDiscriminator()).to(device)  # real vs fake monet
    D_B = init_weights(PatchGANDiscriminator()).to(device)  # real vs fake photo
    ema_A2B = EMA(G_A2B, args.ema) if args.ema > 0 else None
    ema_B2A = EMA(G_B2A, args.ema) if args.ema > 0 else None

    total_params = sum(num_params(m) for m in [G_A2B, G_B2A, D_A, D_B])
    print(f"Total parameters (G_A2B+G_B2A+D_A+D_B): {total_params:,}")

    criterion_gan = nn.MSELoss()   # LSGAN
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    opt_G = torch.optim.Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=args.lr, betas=(args.beta1, 0.999)
    )
    opt_D = torch.optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()), lr=args.lr, betas=(args.beta1, 0.999)
    )
    total_epochs = args.n_epochs + args.n_epochs_decay
    sched_G = torch.optim.lr_scheduler.LambdaLR(
        opt_G, lr_lambda=lambda e: lambda_lr(e, args.n_epochs, args.n_epochs_decay)
    )
    sched_D = torch.optim.lr_scheduler.LambdaLR(
        opt_D, lr_lambda=lambda e: lambda_lr(e, args.n_epochs, args.n_epochs_decay)
    )
    scaler_G = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_D = torch.amp.GradScaler("cuda", enabled=use_amp)

    pool_fake_A = ImagePool(args.pool_size)  # fake "monet" (domain A style)
    pool_fake_B = ImagePool(args.pool_size)  # fake "photo" (domain B style)

    history = {"loss_G": [], "loss_D": [], "loss_cycle": [], "loss_identity": [], "loss_gan": []}
    grad_norms, nan_events = [], 0
    n_images_seen, train_time, step, start_epoch = 0, 0.0, 0, 0

    # ---------------- resume ----------------
    last_path = os.path.join(args.ckpt_dir, "last.pt")
    resume_path = last_path if args.resume_from == "auto" else args.resume_from
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        G_A2B.load_state_dict(ckpt["G_A2B"]); G_B2A.load_state_dict(ckpt["G_B2A"])
        D_A.load_state_dict(ckpt["D_A"]); D_B.load_state_dict(ckpt["D_B"])
        start_epoch = ckpt["epoch"] + 1
        if "opt_G" in ckpt:  # full training state (written every epoch by this script)
            opt_G.load_state_dict(ckpt["opt_G"]); opt_D.load_state_dict(ckpt["opt_D"])
            sched_G.load_state_dict(ckpt["sched_G"]); sched_D.load_state_dict(ckpt["sched_D"])
            scaler_G.load_state_dict(ckpt["scaler_G"]); scaler_D.load_state_dict(ckpt["scaler_D"])
            if ema_A2B and "ema_A2B" in ckpt:
                ema_A2B.shadow.load_state_dict(ckpt["ema_A2B"]); ema_B2A.shadow.load_state_dict(ckpt["ema_B2A"])
            history, grad_norms, nan_events = ckpt["history"], ckpt["grad_norms"], ckpt["nan_events"]
            n_images_seen, train_time, step = ckpt["n_images_seen"], ckpt["train_time"], ckpt["step"]
            random.setstate(ckpt["py_rng"]); np.random.set_state(ckpt["np_rng"])
            torch.set_rng_state(ckpt["torch_rng"].cpu())
            if device.type == "cuda" and ckpt.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu())
            print(f"Resumed full state from {resume_path}: continuing at epoch {start_epoch}")
        else:  # v1 checkpoint: weights only
            for _ in range(start_epoch):
                sched_G.step(); sched_D.step()
            print(f"Resumed weights from {resume_path}: continuing at epoch {start_epoch} "
                  f"(optimizer state is fresh, not restored)")

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log_f = open(log_path, "a" if start_epoch > 0 else "w")  # raw, unedited log -- evidence trail

    # ---------------- checkpoint scoring with the TA's notebook ----------------
    scores_path = os.path.join(args.out_dir, "checkpoint_scores.csv")
    score_rows = list(csv.DictReader(open(scores_path))) if (start_epoch > 0 and os.path.exists(scores_path)) else []
    best = {"score": min((float(r["score"]) for r in score_rows), default=float("inf"))}
    best_dir = {}  # per direction: (fid, mifid, label)
    for r in score_rows:
        for d in ("b2a", "a2b"):
            if float(r[f"fid_{d}"]) < best_dir.get(d, (float("inf"),))[0]:
                best_dir[d] = (float(r[f"fid_{d}"]), float(r[f"mifid_{d}"]), f"epoch {r['epoch']} {r['weights']}")
    ta = ta_eval.load_ta_functions(args.ta_notebook, device) if args.score_last > 0 else None
    monet_files, photo_files = list_images(args.data_dir_a), list_images(args.data_dir_b)

    def save_combo():
        if "b2a" not in best_dir or "a2b" not in best_dir:
            return
        g_b2a = torch.load(os.path.join(args.ckpt_dir, "best_G_B2A.pt"), map_location="cpu", weights_only=True)
        g_a2b = torch.load(os.path.join(args.ckpt_dir, "best_G_A2B.pt"), map_location="cpu", weights_only=True)
        torch.save({"G_A2B": g_a2b, "G_B2A": g_b2a, "config": vars(args),
                    "selected": {"photo_to_monet": best_dir["b2a"][2], "monet_to_photo": best_dir["a2b"][2]}},
                   os.path.join(args.ckpt_dir, "best_combo.pt"))
        (fb, mb, lb), (fa, ma, la) = best_dir["b2a"], best_dir["a2b"]
        print(f"  -> combo: photo->monet from {lb} (FID {fb:.3f}) + monet->photo from {la} (FID {fa:.3f}) "
              f"= {((fb + fa) / 2 + (mb + ma) / 2) / 2:.4f}", flush=True)

    def score_epoch(epoch):
        kinds = [("raw", G_A2B, G_B2A)]
        if ema_A2B:
            kinds.append(("ema", ema_A2B.shadow, ema_B2A.shadow))
        for kind, g_a2b, g_b2a in kinds:
            r = ta_eval.score_generators(g_a2b, g_b2a, monet_files, photo_files, args.data_dir_a, args.data_dir_b,
                                         ta, device, views_a2b=args.views_a2b, img_size=args.img_size)
            r = {"epoch": epoch, "weights": kind, **{k: round(v, 4) for k, v in r.items()}}
            score_rows.append(r)
            with open(scores_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(r)); w.writeheader(); w.writerows(score_rows)
            print(f"SCORE {r}", flush=True)
            label, improved = f"epoch {epoch} {kind}", False
            if r["fid_b2a"] < best_dir.get("b2a", (float("inf"),))[0]:
                best_dir["b2a"] = (r["fid_b2a"], r["mifid_b2a"], label)
                torch.save(g_b2a.state_dict(), os.path.join(args.ckpt_dir, "best_G_B2A.pt")); improved = True
            if r["fid_a2b"] < best_dir.get("a2b", (float("inf"),))[0]:
                best_dir["a2b"] = (r["fid_a2b"], r["mifid_a2b"], label)
                torch.save(g_a2b.state_dict(), os.path.join(args.ckpt_dir, "best_G_A2B.pt")); improved = True
            if improved:
                save_combo()
            if r["score"] < best["score"]:
                best.update(score=r["score"], label=label)
                torch.save({"G_A2B": g_a2b.state_dict(), "G_B2A": g_b2a.state_dict(), "epoch": epoch,
                            "weights": kind, "config": vars(args)}, os.path.join(args.ckpt_dir, "best.pt"))
                print(f"  -> new best {r['score']} ({label}), saved best.pt", flush=True)
            g_a2b.train(kind == "raw"); g_b2a.train(kind == "raw")

    # ---------------- train ----------------
    for epoch in range(start_epoch, total_epochs):
        epoch_losses = {k: 0.0 for k in history}
        n_batches = 0
        t_epoch = time.time()
        G_A2B.train(); G_B2A.train(); D_A.train(); D_B.train()

        for batch in loader:
            if args.max_steps and step >= args.max_steps:
                break
            if n_batches % 100 == 0:
                cool_down(args.max_temp, args.resume_temp)
            real_A = batch["A"].to(device, non_blocking=True)  # monet-style
            real_B = batch["B"].to(device, non_blocking=True)  # photo-style
            b = real_A.size(0)

            # ---------------- Generators ----------------
            opt_G.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                # identity loss: G_B2A(real_A) should look like real_A already (color preservation)
                idt_A = G_B2A(real_A)
                loss_idt_A = criterion_identity(idt_A, real_A) * args.lambda_cycle * args.lambda_identity / 10.0
                idt_B = G_A2B(real_B)
                loss_idt_B = criterion_identity(idt_B, real_B) * args.lambda_cycle * args.lambda_identity / 10.0

                fake_B = G_A2B(real_A)  # translate a domain-A (monet) image into domain-B (photo) style
                fake_A = G_B2A(real_B)  # translate a domain-B (photo) image into domain-A (monet) style
                                         # -- this is the direction the Kaggle submission needs

                pred_fake_A = D_A(diff_augment(fake_A, args.diffaug))
                loss_gan_B2A = criterion_gan(pred_fake_A, torch.ones_like(pred_fake_A))
                pred_fake_B = D_B(diff_augment(fake_B, args.diffaug))
                loss_gan_A2B = criterion_gan(pred_fake_B, torch.ones_like(pred_fake_B))

                rec_A = G_B2A(fake_B)
                loss_cycle_A = criterion_cycle(rec_A, real_A) * args.lambda_cycle
                rec_B = G_A2B(fake_A)
                loss_cycle_B = criterion_cycle(rec_B, real_B) * args.lambda_cycle

                loss_G = (loss_gan_A2B + loss_gan_B2A + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B)
            scaler_G.scale(loss_G).backward()
            scaler_G.unscale_(opt_G)
            grad_norm_G = torch.nn.utils.clip_grad_norm_(
                list(G_A2B.parameters()) + list(G_B2A.parameters()), max_norm=args.grad_clip or float("inf")
            ).item()
            scaler_G.step(opt_G)
            scaler_G.update()
            if ema_A2B:
                ema_A2B.update(G_A2B); ema_B2A.update(G_B2A)

            # ---------------- Discriminators ----------------
            opt_D.zero_grad(set_to_none=True)
            fake_A_pooled = pool_fake_A.query(fake_A.detach())
            fake_B_pooled = pool_fake_B.query(fake_B.detach())
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred_real_A = D_A(diff_augment(real_A, args.diffaug))
                pred_fake_A_d = D_A(diff_augment(fake_A_pooled, args.diffaug))
                loss_D_A = 0.5 * (
                    criterion_gan(pred_real_A, torch.full_like(pred_real_A, args.real_label))
                    + criterion_gan(pred_fake_A_d, torch.zeros_like(pred_fake_A_d))
                )
                pred_real_B = D_B(diff_augment(real_B, args.diffaug))
                pred_fake_B_d = D_B(diff_augment(fake_B_pooled, args.diffaug))
                loss_D_B = 0.5 * (
                    criterion_gan(pred_real_B, torch.full_like(pred_real_B, args.real_label))
                    + criterion_gan(pred_fake_B_d, torch.zeros_like(pred_fake_B_d))
                )
                loss_D = loss_D_A + loss_D_B
            scaler_D.scale(loss_D).backward()
            scaler_D.unscale_(opt_D)
            grad_norm_D = torch.nn.utils.clip_grad_norm_(
                list(D_A.parameters()) + list(D_B.parameters()), max_norm=args.grad_clip or float("inf")
            ).item()
            scaler_D.step(opt_D)
            scaler_D.update()

            is_nan = not all(
                torch.isfinite(v).all().item() for v in [loss_G, loss_D]
            )
            if is_nan:
                nan_events += 1
            if np.isfinite(grad_norm_G) and np.isfinite(grad_norm_D):
                grad_norms.append(grad_norm_G + grad_norm_D)

            n_images_seen += b
            step += 1
            n_batches += 1
            epoch_losses["loss_G"] += loss_G.item()
            epoch_losses["loss_D"] += loss_D.item()
            epoch_losses["loss_cycle"] += (loss_cycle_A + loss_cycle_B).item()
            epoch_losses["loss_identity"] += (loss_idt_A + loss_idt_B).item()
            epoch_losses["loss_gan"] += (loss_gan_A2B + loss_gan_B2A).item()

            if step % args.log_every == 0:
                elapsed = train_time + time.time() - t_epoch
                rec = {
                    "step": step, "epoch": epoch,
                    "loss_G": loss_G.item(), "loss_D": loss_D.item(),
                    "loss_cycle": (loss_cycle_A + loss_cycle_B).item(),
                    "loss_identity": (loss_idt_A + loss_idt_B).item(),
                    "loss_gan": (loss_gan_A2B + loss_gan_B2A).item(),
                    "grad_norm_G": grad_norm_G, "grad_norm_D": grad_norm_D,
                    "lr": sched_G.get_last_lr()[0],
                    "images_per_sec": n_images_seen / max(elapsed, 1e-8),
                    "elapsed_sec": elapsed, "nan_or_inf": is_nan,
                }
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()

        for k in history:
            history[k].append(epoch_losses[k] / max(n_batches, 1))
        sched_G.step()
        sched_D.step()
        train_time += time.time() - t_epoch
        print(f"[epoch {epoch}] loss_G={history['loss_G'][-1]:.3f} loss_D={history['loss_D'][-1]:.3f} "
              f"cycle={history['loss_cycle'][-1]:.3f} identity={history['loss_identity'][-1]:.3f} "
              f"lr={sched_G.get_last_lr()[0]:.2e} time={time.time()-t_epoch:.1f}s", flush=True)

        if args.score_last > 0 and epoch >= total_epochs - args.score_last:
            score_epoch(epoch)

        state = {"G_A2B": G_A2B.state_dict(), "G_B2A": G_B2A.state_dict(),
                 "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
                 "epoch": epoch, "config": vars(args)}
        if args.save_every_epoch and (epoch + 1) % args.save_every_epoch == 0:
            torch.save(state, os.path.join(args.ckpt_dir, f"ckpt_epoch{epoch}.pt"))
        state.update({
            "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
            "sched_G": sched_G.state_dict(), "sched_D": sched_D.state_dict(),
            "scaler_G": scaler_G.state_dict(), "scaler_D": scaler_D.state_dict(),
            "history": history, "grad_norms": grad_norms, "nan_events": nan_events,
            "n_images_seen": n_images_seen, "train_time": train_time, "step": step,
            "py_rng": random.getstate(), "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
        })
        if ema_A2B:
            state.update({"ema_A2B": ema_A2B.shadow.state_dict(), "ema_B2A": ema_B2A.shadow.state_dict()})
        torch.save(state, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)  # written atomically so a crash cannot corrupt it

        if args.max_steps and step >= args.max_steps:
            print(f"[epoch {epoch}] stopped early at max_steps={args.max_steps}")
            break

    log_f.close()

    final = {"G_A2B": G_A2B.state_dict(), "G_B2A": G_B2A.state_dict(),
             "D_A": D_A.state_dict(), "D_B": D_B.state_dict(), "config": vars(args)}
    if ema_A2B:
        final.update({"ema_A2B": ema_A2B.shadow.state_dict(), "ema_B2A": ema_B2A.shadow.state_dict()})
    torch.save(final, os.path.join(args.ckpt_dir, "ckpt_final.pt"))

    # ---- loss curves ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history["loss_G"], label="G total")
    axes[0].plot(history["loss_D"], label="D total")
    axes[0].plot(history["loss_gan"], label="GAN (adv) component")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title("Generator / Discriminator loss")

    axes[1].plot(history["loss_cycle"], label="cycle-consistency")
    axes[1].plot(history["loss_identity"], label="identity")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_title("Cycle-consistency / identity loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss_curves.png"), dpi=150)
    plt.close()

    if HAS_RESOURCE:
        peak_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    else:
        try:
            import psutil
            peak_mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 2)
        except ImportError:
            peak_mem_mb = None
    metrics = {
        "device": str(device),
        "total_parameter_count": total_params,
        "epochs_trained": len(history["loss_G"]),
        "steps": step,
        "final_loss_G": history["loss_G"][-1] if history["loss_G"] else None,
        "final_loss_D": history["loss_D"][-1] if history["loss_D"] else None,
        "final_loss_cycle": history["loss_cycle"][-1] if history["loss_cycle"] else None,
        "final_loss_identity": history["loss_identity"][-1] if history["loss_identity"] else None,
        "mean_grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else None,
        "max_grad_norm": max(grad_norms) if grad_norms else None,
        "nan_or_inf_events": nan_events,
        "total_training_time_sec": train_time,
        "avg_images_per_sec": n_images_seen / max(train_time, 1e-8),
        "peak_memory_mb": peak_mem_mb,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None,
        "best_ta_score": best["score"] if best["score"] < float("inf") else None,
        "best_ta_checkpoint": best.get("label"),
        "loss_history": history,
    }
    with open(os.path.join(args.out_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(json.dumps({k: v for k, v in metrics.items() if k != "loss_history"}, indent=2))


if __name__ == "__main__":
    main()
