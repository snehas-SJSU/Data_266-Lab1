"""
Task 3.1(2,3,5) - CycleGAN training: adversarial (LSGAN) + cycle-consistency
+ identity loss, image pool for discriminator stability, linear LR decay.
Logs everything needed for Task 3's "analyse training behaviour" requirement:
raw per-step JSONL log (do not hand-edit -- evidence trail), gen/disc loss
curves, cycle/identity loss curves, gradient norms + NaN/Inf count, images/sec,
peak memory, total training time, parameter counts.
"""
import argparse
import json
import os
import time

try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import ResnetGenerator, PatchGANDiscriminator, init_weights, num_params
from dataset import UnpairedImageDataset
from utils import ImagePool, lambda_lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir_a", default="../../data/monet_jpg", help="domain A (e.g. Monet paintings)")
    ap.add_argument("--data_dir_b", default="../../data/photo_jpg", help="domain B (e.g. photos)")
    ap.add_argument("--out_dir", default="../outputs")
    ap.add_argument("--ckpt_dir", default="../checkpoints")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--n_epochs", type=int, default=10, help="epochs at full LR")
    ap.add_argument("--n_epochs_decay", type=int, default=10, help="epochs of linear LR decay to 0")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--lambda_cycle", type=float, default=10.0)
    ap.add_argument("--lambda_identity", type=float, default=5.0)
    ap.add_argument("--n_blocks", type=int, default=9, help="generator residual blocks (6 for 128px, 9 for 256px)")
    ap.add_argument("--pool_size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every_epoch", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None, help="optional hard cap for smoke tests")
    ap.add_argument("--resume_from", default=None, help="path to a ckpt_epoch*.pt to resume from (reloads G/D weights, continues epoch count + LR schedule; optimizer momentum is NOT restored, so expect one noisy epoch right after resuming)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = UnpairedImageDataset(args.data_dir_a, args.data_dir_b, args.img_size, train=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    print(f"Domain A (monet-style): {len(dataset.files_a)} images | Domain B (photo-style): {len(dataset.files_b)} images")

    G_A2B = init_weights(ResnetGenerator(n_blocks=args.n_blocks)).to(device)  # photo -> monet
    G_B2A = init_weights(ResnetGenerator(n_blocks=args.n_blocks)).to(device)  # monet -> photo
    D_A = init_weights(PatchGANDiscriminator()).to(device)  # real vs fake monet
    D_B = init_weights(PatchGANDiscriminator()).to(device)  # real vs fake photo

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

    pool_fake_A = ImagePool(args.pool_size)  # fake "monet" (domain A style)
    pool_fake_B = ImagePool(args.pool_size)  # fake "photo" (domain B style)

    start_epoch = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        G_A2B.load_state_dict(ckpt["G_A2B"])
        G_B2A.load_state_dict(ckpt["G_B2A"])
        D_A.load_state_dict(ckpt["D_A"])
        D_B.load_state_dict(ckpt["D_B"])
        start_epoch = ckpt["epoch"] + 1
        for _ in range(start_epoch):
            sched_G.step()
            sched_D.step()
        print(f"Resumed from {args.resume_from}: continuing at epoch {start_epoch} (optimizer state is fresh, not restored)")

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log_f = open(log_path, "a" if args.resume_from else "w")  # raw, unedited log -- evidence trail

    history = {"loss_G": [], "loss_D": [], "loss_cycle": [], "loss_identity": [], "loss_gan": []}
    grad_norms, nan_events = [], 0
    n_images_seen = 0
    t_start = time.time()
    step = 0

    for epoch in range(start_epoch, total_epochs):
        epoch_losses = {k: 0.0 for k in history}
        n_batches = 0
        t_epoch = time.time()

        for batch in loader:
            if args.max_steps and step >= args.max_steps:
                break
            real_A = batch["A"].to(device)  # monet-style
            real_B = batch["B"].to(device)  # photo-style
            b = real_A.size(0)

            # ---------------- Generators ----------------
            opt_G.zero_grad(set_to_none=True)

            # identity loss: G_A2B(real_A) should look like real_A already (color preservation)
            idt_A = G_B2A(real_A)
            loss_idt_A = criterion_identity(idt_A, real_A) * args.lambda_cycle * args.lambda_identity / 10.0
            idt_B = G_A2B(real_B)
            loss_idt_B = criterion_identity(idt_B, real_B) * args.lambda_cycle * args.lambda_identity / 10.0

            fake_B = G_A2B(real_A)  # translate a domain-A (monet) image into domain-B (photo) style
            fake_A = G_B2A(real_B)  # translate a domain-B (photo) image into domain-A (monet) style
                                     # -- this is the direction the Kaggle submission needs

            pred_fake_A = D_A(fake_A)
            loss_gan_B2A = criterion_gan(pred_fake_A, torch.ones_like(pred_fake_A))
            pred_fake_B = D_B(fake_B)
            loss_gan_A2B = criterion_gan(pred_fake_B, torch.ones_like(pred_fake_B))

            rec_A = G_B2A(fake_B)
            loss_cycle_A = criterion_cycle(rec_A, real_A) * args.lambda_cycle
            rec_B = G_A2B(fake_A)
            loss_cycle_B = criterion_cycle(rec_B, real_B) * args.lambda_cycle

            loss_G = (loss_gan_A2B + loss_gan_B2A + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B)
            loss_G.backward()
            grad_norm_G = torch.nn.utils.clip_grad_norm_(
                list(G_A2B.parameters()) + list(G_B2A.parameters()), max_norm=10.0
            ).item()
            opt_G.step()

            # ---------------- Discriminators ----------------
            opt_D.zero_grad(set_to_none=True)

            fake_A_pooled = pool_fake_A.query(fake_A.detach())
            pred_real_A = D_A(real_A)
            pred_fake_A_d = D_A(fake_A_pooled)
            loss_D_A = 0.5 * (
                criterion_gan(pred_real_A, torch.ones_like(pred_real_A))
                + criterion_gan(pred_fake_A_d, torch.zeros_like(pred_fake_A_d))
            )

            fake_B_pooled = pool_fake_B.query(fake_B.detach())
            pred_real_B = D_B(real_B)
            pred_fake_B_d = D_B(fake_B_pooled)
            loss_D_B = 0.5 * (
                criterion_gan(pred_real_B, torch.ones_like(pred_real_B))
                + criterion_gan(pred_fake_B_d, torch.zeros_like(pred_fake_B_d))
            )

            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            grad_norm_D = torch.nn.utils.clip_grad_norm_(
                list(D_A.parameters()) + list(D_B.parameters()), max_norm=10.0
            ).item()
            opt_D.step()

            is_nan = not all(
                torch.isfinite(v).all().item() for v in [loss_G, loss_D]
            )
            if is_nan:
                nan_events += 1
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
                elapsed = time.time() - t_start
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

        if args.max_steps and step >= args.max_steps:
            for k in history:
                history[k].append(epoch_losses[k] / max(n_batches, 1))
            torch.save(
                {"G_A2B": G_A2B.state_dict(), "G_B2A": G_B2A.state_dict(),
                 "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
                 "epoch": epoch, "config": vars(args)},
                os.path.join(args.ckpt_dir, f"ckpt_epoch{epoch}.pt"),
            )
            print(f"[epoch {epoch}] stopped early at max_steps={args.max_steps} "
                  f"(partial epoch, {n_batches} batches) "
                  f"loss_G={history['loss_G'][-1]:.3f} loss_D={history['loss_D'][-1]:.3f}")
            break

        for k in history:
            history[k].append(epoch_losses[k] / max(n_batches, 1))
        sched_G.step()
        sched_D.step()
        print(f"[epoch {epoch}] loss_G={history['loss_G'][-1]:.3f} loss_D={history['loss_D'][-1]:.3f} "
              f"cycle={history['loss_cycle'][-1]:.3f} identity={history['loss_identity'][-1]:.3f} "
              f"time={time.time()-t_epoch:.1f}s")

        if (epoch + 1) % args.save_every_epoch == 0:
            torch.save(
                {"G_A2B": G_A2B.state_dict(), "G_B2A": G_B2A.state_dict(),
                 "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
                 "epoch": epoch, "config": vars(args)},
                os.path.join(args.ckpt_dir, f"ckpt_epoch{epoch}.pt"),
            )

    total_time = time.time() - t_start
    log_f.close()

    torch.save(
        {"G_A2B": G_A2B.state_dict(), "G_B2A": G_B2A.state_dict(),
         "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
         "config": vars(args)},
        os.path.join(args.ckpt_dir, "ckpt_final.pt"),
    )

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
        "final_loss_G": history["loss_G"][-1] if history["loss_G"] else None,
        "final_loss_D": history["loss_D"][-1] if history["loss_D"] else None,
        "final_loss_cycle": history["loss_cycle"][-1] if history["loss_cycle"] else None,
        "final_loss_identity": history["loss_identity"][-1] if history["loss_identity"] else None,
        "mean_grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else None,
        "max_grad_norm": max(grad_norms) if grad_norms else None,
        "nan_or_inf_events": nan_events,
        "total_training_time_sec": total_time,
        "avg_images_per_sec": n_images_seen / total_time,
        "peak_memory_mb": peak_mem_mb,
        "loss_history": history,
    }
    with open(os.path.join(args.out_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(json.dumps({k: v for k, v in metrics.items() if k != "loss_history"}, indent=2))


if __name__ == "__main__":
    main()
