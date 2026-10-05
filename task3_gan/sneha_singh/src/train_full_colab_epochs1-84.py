"""Full CycleGAN training from scratch with Sneha's submitted recipe, plus one changed setting (default: --ngf 64).

Recipe (src/part3_cyclegan.ipynb, submitted run): ResNet-9 generators with nearest upsampling, PatchGAN discriminators,
LSGAN + cycle L1 (10) + identity (0.5 x 10), DiffAugment (color, translation, cutout) on everything D sees,
real labels 0.9, pool 50, Adam(2e-4, beta1 0.5), batch 4, 800 steps/epoch with 3,200 photos resampled each epoch,
50 epochs constant lr + 50 epochs linear decay, AMP.

Auto-resume: the full training state is saved to --out/last.pt after every epoch; rerunning the same command
continues from the last finished epoch. The last --score_last epochs are scored with the TA's method
(first 300 photos -> Monet single pass, first 300 Monets -> photo 6-view average, Inception FID / MiFID) and the
best-scoring epoch is kept as --out/best.pt.

Colab:  python train_full.py --name ngf64 --ngf 64 --out /content/drive/MyDrive/part3_full
Smoke:  python train_full.py --name smoke --smoke --out /tmp/smoke_full
"""

import argparse
import copy
import csv
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--ngf", type=int, default=64, help="generator base filters (Sneha's submission: 32)")
ap.add_argument("--epochs_const", type=int, default=50)
ap.add_argument("--epochs_decay", type=int, default=50)
ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--photos_per_epoch", type=int, default=3200)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--lr_d", type=float, default=2e-4)
ap.add_argument("--lambda_cycle", type=float, default=10.0)
ap.add_argument("--lambda_identity", type=float, default=0.5, help="multiplied by lambda_cycle (0.5 -> weight 5)")
ap.add_argument("--smooth", type=float, default=0.9)
ap.add_argument("--ema", type=float, default=0.999, help="EMA decay of generator weights per step (0 = off)")
ap.add_argument("--score_last", type=int, default=15, help="score the last N epochs with the TA method")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--seed", type=int, default=670)
ap.add_argument("--max_temp", type=int, default=95, help="pause when GPU reaches this temperature (laptop: 84)")
ap.add_argument("--resume_temp", type=int, default=74)
ap.add_argument("--repo", default="/content/Data_266-Lab1")
ap.add_argument("--out", default="/content/drive/MyDrive/part3_full")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke:
    args.epochs_const, args.epochs_decay, args.steps, args.photos_per_epoch, args.score_last = 1, 1, 3, 12, 2
if os.name == "nt":
    args.workers = 0  # Windows: no DataLoader worker processes

N_EPOCHS = args.epochs_const + args.epochs_decay
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp = dev.type == "cuda"
torch.backends.cudnn.benchmark = True

M = Path(args.repo) / "task3_gan" / "sneha_singh"
sys.path.insert(0, str(M))
import make_submission as ms  # translate / views / to_pil of the submitted outputs

sc = ms.sc  # score_checkpoints: generator class + TA FID/MiFID
OUT = Path(args.out) / args.name
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "train.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return None


def cool_down():
    t = gpu_temp()
    if t is None or t < args.max_temp:
        return
    log(f"  [thermal] GPU at {t} C, pausing until {args.resume_temp} C ...")
    while t is not None and t > args.resume_temp:
        time.sleep(20)
        t = gpu_temp()
    log(f"  [thermal] cooled to {t} C, resuming")


monet_all = sc.ev._list_images(sc.DATA / "monet_jpg")
photo_all = sc.ev._list_images(sc.DATA / "photo_jpg")
assert len(monet_all) == 300 and len(photo_all) == 7038, (len(monet_all), len(photo_all))


# ---------------- data (same transforms as the notebook) ----------------
class Images(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.tf = transforms.Compose([
            transforms.Resize(286, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(256), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def loader(paths, persistent):
    return DataLoader(Images(paths), batch_size=4, shuffle=True, drop_last=True, num_workers=args.workers,
                      pin_memory=amp, persistent_workers=persistent and args.workers > 0)


monet_loader = loader(monet_all, persistent=True)


def photo_loader(epoch):
    """Resample photos_per_epoch photos each epoch (notebook: random.Random(SEED + epoch))."""
    paths = list(photo_all)
    if 0 < args.photos_per_epoch < len(paths):
        paths = random.Random(args.seed + int(epoch)).sample(paths, args.photos_per_epoch)
    return loader(paths, persistent=False)


# ---------------- discriminator, pool, DiffAugment (same as the notebook) ----------------
class PatchD(nn.Module):
    def __init__(self, ndf=64, n_layers=3):
        super().__init__()
        seq = [nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * prev, ndf * mult, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(ndf * mult, affine=False, track_running_stats=False), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * prev, ndf * mult, 4, 1, 1, bias=False),
                nn.InstanceNorm2d(ndf * mult, affine=False, track_running_stats=False), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * mult, 1, 4, 1, 1)]
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)


class Pool:
    def __init__(self, n):
        self.n, self.items = n, []

    def query(self, imgs):
        out = []
        for im in imgs:
            im = im.detach().unsqueeze(0)
            if len(self.items) < self.n:
                self.items.append(im); out.append(im)
            elif random.random() > 0.5:
                j = random.randint(0, self.n - 1)
                out.append(self.items[j].clone()); self.items[j] = im
            else:
                out.append(im)
        return torch.cat(out, 0)


def _bright(x): return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
def _sat(x):
    m = x.mean(dim=1, keepdim=True)
    return (x - m) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + m
def _contrast(x):
    m = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - m) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + m
def _translation(x, r=0.125):
    sx, sy = int(x.size(2) * r + 0.5), int(x.size(3) * r + 0.5)
    tx = torch.randint(-sx, sx + 1, size=[x.size(0), 1, 1], device=x.device)
    ty = torch.randint(-sy, sy + 1, size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(x.size(0), device=x.device), torch.arange(x.size(2), device=x.device),
                                torch.arange(x.size(3), device=x.device), indexing="ij")
    gx = torch.clamp(gx + tx + 1, 0, x.size(2) + 1); gy = torch.clamp(gy + ty + 1, 0, x.size(3) + 1)
    return F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0]).permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)
def _cutout(x, r=0.5):
    cs = int(x.size(2) * r + 0.5), int(x.size(3) * r + 0.5)
    ox = torch.randint(0, x.size(2) + (1 - cs[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    oy = torch.randint(0, x.size(3) + (1 - cs[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(x.size(0), device=x.device), torch.arange(cs[0], device=x.device),
                                torch.arange(cs[1], device=x.device), indexing="ij")
    gx = torch.clamp(gx + ox - cs[0] // 2, 0, x.size(2) - 1); gy = torch.clamp(gy + oy - cs[1] // 2, 0, x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)
_AUG = [_bright, _sat, _contrast, _translation, _cutout]
def diff_aug(x):
    for f in _AUG:
        x = f(x)
    return x


# ---------------- models, optimizers, schedule ----------------
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
G_AB = sc.ResnetGenerator("nearest", ngf=args.ngf, n_blocks=9).to(dev)  # photo -> Monet
G_BA = sc.ResnetGenerator("nearest", ngf=args.ngf, n_blocks=9).to(dev)  # Monet -> photo
D_A, D_B = PatchD(64).to(dev), PatchD(64).to(dev)                      # Monet / photo critics
params = sum(p.numel() for m in (G_AB, G_BA, D_A, D_B) for p in m.parameters())
E_AB, E_BA = copy.deepcopy(G_AB).eval(), copy.deepcopy(G_BA).eval()  # EMA generators
for p_ in list(E_AB.parameters()) + list(E_BA.parameters()):
    p_.requires_grad_(False)


@torch.no_grad()
def ema_update():
    for e, g in ((E_AB, G_AB), (E_BA, G_BA)):
        for pe, pg in zip(e.parameters(), g.parameters()):
            pe.mul_(args.ema).add_(pg.detach(), alpha=1 - args.ema)

opt_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=args.lr, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=args.lr_d, betas=(0.5, 0.999))


def lr_lambda(e):  # notebook: stepped once per epoch, e = epochs finished so far
    return 1.0 if e < args.epochs_const else max(0.0, 1.0 - (e - args.epochs_const) / max(1, args.epochs_decay))


sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)
scaler = torch.amp.GradScaler("cuda", enabled=amp)
mse, l1 = nn.MSELoss(), nn.L1Loss()
pool_A, pool_B = Pool(50), Pool(50)
L_C, L_I, SM = args.lambda_cycle, args.lambda_identity * args.lambda_cycle, args.smooth

# ---------------- resume ----------------
start_epoch, train_time = 1, 0.0
last = OUT / "last.pt"
if last.exists():
    st = torch.load(last, map_location=dev, weights_only=False)
    for k, m in [("G_AB", G_AB), ("G_BA", G_BA), ("D_A", D_A), ("D_B", D_B), ("E_AB", E_AB), ("E_BA", E_BA)]:
        m.load_state_dict(st[k])
    opt_G.load_state_dict(st["opt_G"]); opt_D.load_state_dict(st["opt_D"])
    sch_G.load_state_dict(st["sch_G"]); sch_D.load_state_dict(st["sch_D"]); scaler.load_state_dict(st["scaler"])
    random.setstate(st["py_rng"]); np.random.set_state(st["np_rng"]); torch.set_rng_state(st["torch_rng"].cpu())
    if amp and st.get("cuda_rng") is not None:
        torch.cuda.set_rng_state(st["cuda_rng"].cpu())
    start_epoch, train_time = st["epoch"] + 1, st["train_time"]
    log(f"RESUMING after epoch {st['epoch']} | lr {opt_G.param_groups[0]['lr']:.2e}")
else:
    log(f"START {vars(args)} | device {dev} | params {params:,}")

# ---------------- scoring (TA method, same pipeline as make_submission.py) ----------------
@torch.no_grad()
def write_dir(G, paths, out, views):
    out.mkdir(parents=True, exist_ok=True)
    was_training = G.training
    G.eval()
    for i, p in enumerate(paths):
        x = sc.TF(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
        ms.to_pil(ms.translate(G, x, views)).save(out / f"{i:05d}.jpg", quality=95)
    G.train(was_training)


def score(g_ab, g_ba):
    rm, rp = monet_all[:300], photo_all[:300]
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        write_dir(g_ab, rp, t / "b2a", ms.SINGLE)
        write_dir(g_ba, rm, t / "a2b", ms.SIX_VIEWS)
        fb, mb = sc.ev.calculate_fid_mifid(rm, sc.ev._list_images(t / "b2a")[:300], dev)
        fa, ma = sc.ev.calculate_fid_mifid(rp, sc.ev._list_images(t / "a2b")[:300], dev)
    fid, mifid = (fa + fb) / 2, (ma + mb) / 2
    return dict(fid_b2a=round(fb, 3), fid_a2b=round(fa, 3), mifid=round(mifid, 4), score=round((fid + mifid) / 2, 4))


scores_csv = OUT / "scores.csv"
rows = list(csv.DictReader(open(scores_csv))) if scores_csv.exists() else []
best_score = min((float(r["score"]) for r in rows), default=float("inf"))

# ---------------- train ----------------
t_run = time.time()
for epoch in range(start_epoch, N_EPOCHS + 1):
    G_AB.train(); G_BA.train(); D_A.train(); D_B.train()
    t_ep = time.time()
    it_m, it_p = iter(monet_loader), iter(photo_loader(epoch))
    sums, nan = np.zeros(4), 0
    for si in range(args.steps):
        if si % 100 == 0:
            cool_down()
        try:
            rA = next(it_m)
        except StopIteration:
            it_m = iter(monet_loader); rA = next(it_m)
        try:
            rB = next(it_p)
        except StopIteration:
            it_p = iter(photo_loader(epoch)); rB = next(it_p)
        rA, rB = rA.to(dev, non_blocking=True), rB.to(dev, non_blocking=True)

        opt_G.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            fB, fA = G_BA(rA), G_AB(rB)
            loss_id = (l1(G_AB(rA), rA) + l1(G_BA(rB), rB)) * L_I
            loss_cyc = (l1(G_AB(fB), rA) + l1(G_BA(fA), rB)) * L_C
            pA, pB = D_A(diff_aug(fA)), D_B(diff_aug(fB))
            loss_gan = mse(pA, torch.ones_like(pA)) + mse(pB, torch.ones_like(pB))
            loss_G = loss_gan + loss_cyc + loss_id
        scaler.scale(loss_G).backward(); scaler.step(opt_G)

        opt_D.zero_grad(set_to_none=True)
        qA, qB = pool_A.query(fA), pool_B.query(fB)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            ra, rb, fa_, fb_ = D_A(diff_aug(rA)), D_B(diff_aug(rB)), D_A(diff_aug(qA)), D_B(diff_aug(qB))
            loss_D = 0.5 * (mse(ra, torch.ones_like(ra) * SM) + mse(fa_, torch.zeros_like(fa_))) + \
                     0.5 * (mse(rb, torch.ones_like(rb) * SM) + mse(fb_, torch.zeros_like(fb_)))
        scaler.scale(loss_D).backward(); scaler.step(opt_D); scaler.update()
        if args.ema > 0:
            ema_update()

        nan += int(not (torch.isfinite(loss_G) and torch.isfinite(loss_D)))
        sums += [loss_G.item(), loss_D.item(), loss_cyc.item(), loss_id.item()]
    sch_G.step(); sch_D.step()
    train_time += time.time() - t_ep
    g, d, c, i = (float(v) for v in sums / args.steps)
    log(f"epoch {epoch}/{N_EPOCHS} g={g:.3f} d={d:.3f} cycle={c:.3f} id={i:.3f} nan={nan} "
        f"lr={opt_G.param_groups[0]['lr']:.2e} epoch_min={(time.time() - t_ep) / 60:.1f} train_h={train_time / 3600:.2f}")

    # score the last epochs with the TA method, keep the best generators
    if epoch > N_EPOCHS - args.score_last:
        kinds = [("raw", G_AB, G_BA)] + ([("ema", E_AB, E_BA)] if args.ema > 0 else [])
        for kind, g_ab, g_ba in kinds:
            r = dict(epoch=epoch, weights=kind, g=round(g, 3), d=round(d, 3), **score(g_ab, g_ba))
            rows.append(r)
            with open(scores_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(r)); w.writeheader(); w.writerows(rows)
            log(f"SCORE {r}")
            if r["score"] < best_score:
                best_score = r["score"]
                torch.save({"G_AB": g_ab.state_dict(), "G_BA": g_ba.state_dict(), "D_A": D_A.state_dict(),
                            "D_B": D_B.state_dict(), "epoch": epoch, "ngf": args.ngf, "weights": kind}, OUT / "best.pt")
                log(f"  -> new best {best_score} (epoch {epoch}, {kind} weights), saved {OUT / 'best.pt'}")

    # full state for auto-resume (written to a temp file first so a crash cannot corrupt it)
    torch.save({"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(), "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
                "E_AB": E_AB.state_dict(), "E_BA": E_BA.state_dict(),
                "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(), "sch_G": sch_G.state_dict(),
                "sch_D": sch_D.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "train_time": train_time,
                "py_rng": random.getstate(), "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if amp else None}, OUT / "last.tmp")
    os.replace(OUT / "last.tmp", last)

peak = torch.cuda.max_memory_allocated() / 2 ** 20 if amp else 0.0
log(f"DONE train_h={train_time / 3600:.2f} peak_memory_mb={peak:.0f}")
for r in sorted(rows, key=lambda r: float(r["score"])):
    log(f"  {r}")
log(f"BEST {best_score}  ->  {OUT / 'best.pt'}")