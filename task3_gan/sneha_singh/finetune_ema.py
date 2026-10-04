"""Part 3 fine-tune + EMA (Sneha Singh).

Continues training the epoch-93 CycleGAN (both generators and both discriminators) at a low
learning rate and keeps an exponential moving average (EMA) of the generator weights.
The EMA generators are scored with the professor evaluation method; the best one is turned into
pred_A2B / pred_B2A and submission.csv.

Colab (Lab-1 already unzipped at /content/Lab-1, Drive mounted):
    !python /content/finetune_ema.py 2>&1 | tee /content/finetune_ema_out.txt
Smoke test (tiny, any machine):
    SMOKE=1 python finetune_ema.py
"""

import copy
import csv
import importlib
import os
import random
import runpy
import shutil
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

SMOKE = os.environ.get("SMOKE") == "1"
LAB = Path(os.environ.get("LAB_ROOT", "/content/Lab-1"))
M = LAB / "task3_gan" / "sneha_singh"
sys.path.insert(0, str(M))
sc = importlib.import_module("score_checkpoints")  # generator definition + professor FID/MiFID

DRIVE = Path("/content/drive/MyDrive/lab1_part3_sneha_finetune")
OUT = Path("/content/ft_ckpts") if not SMOKE else Path(tempfile.mkdtemp())
LOG = LAB / "reproducibility" / "raw_logs" / "sneha_singh" / "task3_gan" / ("train_finetune_ema.log" if not SMOKE else "smoke_ft.log")

CFG = dict(
    epochs=20, steps_per_epoch=800, batch=4, lr=5e-5, beta1=0.5, lambda_cycle=10.0, lambda_identity=0.5,
    label_smoothing=0.9, pool=50, ema_decay=0.999, diffaugment="color,translation,cutout",
    ngf=32, ndf=64, n_res_blocks=9, load=286, img=256, num_workers=4, score_last=8, seed=670,
)
if SMOKE:
    CFG.update(epochs=2, steps_per_epoch=2, num_workers=0, score_last=1)
N_EVAL = 300 if not SMOKE else 16

random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp = dev.type == "cuda"
print("device:", dev, "| smoke:", SMOKE, "| config:", CFG, flush=True)


# ---------------- data ----------------
class Images(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.tf = transforms.Compose([
            transforms.Resize(CFG["load"], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(CFG["img"]), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def forever(loader):
    while True:
        for x in loader:
            yield x


monet_all = sc.ev._list_images(sc.DATA / "monet_jpg")
photo_all = sc.ev._list_images(sc.DATA / "photo_jpg")
train_m = monet_all if not SMOKE else monet_all[:8]
train_p = photo_all if not SMOKE else photo_all[:8]
kw = dict(batch_size=CFG["batch"], shuffle=True, drop_last=True, num_workers=CFG["num_workers"],
          persistent_workers=CFG["num_workers"] > 0)
it_m, it_p = forever(DataLoader(Images(train_m), **kw)), forever(DataLoader(Images(train_p), **kw))


# ---------------- models (same as the notebook) ----------------
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
_AUG = {"color": [_bright, _sat, _contrast], "translation": [_translation], "cutout": [_cutout]}
def diff_aug(x):
    for p in CFG["diffaugment"].split(","):
        for f in _AUG[p]:
            x = f(x)
    return x


# ---------------- load epoch 93 ----------------
src = M / "checkpoints" / "best.pt"
alt = Path("/content/drive/MyDrive/lab1_part3_sneha_light/best_epoch_093.pt")
st = torch.load(src, map_location="cpu", weights_only=False)
if st.get("epoch") != 93 and alt.exists():
    src, st = alt, torch.load(alt, map_location="cpu", weights_only=False)
assert st.get("epoch") == 93, f"need the epoch-93 checkpoint, got epoch {st.get('epoch')} from {src}"
print("starting from", src, "epoch", st["epoch"], flush=True)

G_AB = sc.ResnetGenerator("nearest", ngf=CFG["ngf"], n_blocks=CFG["n_res_blocks"]).to(dev)  # photo -> Monet
G_BA = sc.ResnetGenerator("nearest", ngf=CFG["ngf"], n_blocks=CFG["n_res_blocks"]).to(dev)  # Monet -> photo
D_A, D_B = PatchD(CFG["ndf"]).to(dev), PatchD(CFG["ndf"]).to(dev)                          # Monet / photo critics
for k, m in [("G_AB", G_AB), ("G_BA", G_BA), ("D_A", D_A), ("D_B", D_B)]:
    m.load_state_dict(st[k])
E_AB, E_BA = copy.deepcopy(G_AB).eval(), copy.deepcopy(G_BA).eval()   # EMA generators
for p in list(E_AB.parameters()) + list(E_BA.parameters()):
    p.requires_grad_(False)

opt_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=CFG["lr"], betas=(CFG["beta1"], 0.999))
opt_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=CFG["lr"], betas=(CFG["beta1"], 0.999))
total = CFG["epochs"] * CFG["steps_per_epoch"]
lin = lambda s: max(0.0, 1.0 - s / total)                       # linear decay to 0 over the whole fine-tune
sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lin)
sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lin)
scaler = torch.amp.GradScaler("cuda", enabled=amp)
mse, l1 = nn.MSELoss(), nn.L1Loss()
pool_A, pool_B = Pool(CFG["pool"]), Pool(CFG["pool"])
L_C, L_I, SM = CFG["lambda_cycle"], CFG["lambda_identity"] * CFG["lambda_cycle"], CFG["label_smoothing"]


@torch.no_grad()
def ema_update():
    d = CFG["ema_decay"]
    for e, g in ((E_AB, G_AB), (E_BA, G_BA)):
        for pe, pg in zip(e.parameters(), g.parameters()):
            pe.mul_(d).add_(pg.detach(), alpha=1 - d)


# ---------------- fine-tune ----------------
OUT.mkdir(parents=True, exist_ok=True)
on_drive = Path("/content/drive/MyDrive").is_dir() and not SMOKE
if on_drive:
    DRIVE.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
with open(LOG, "w") as f:
    f.write(f"member=sneha_singh run=finetune_ema start=epoch93 device={dev} config={CFG}\n")
t0 = time.time()
if dev.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
for epoch in range(1, CFG["epochs"] + 1):
    G_AB.train(); G_BA.train(); D_A.train(); D_B.train()
    sums = np.zeros(4); nan = 0
    for _ in range(CFG["steps_per_epoch"]):
        rA, rB = next(it_m).to(dev, non_blocking=True), next(it_p).to(dev, non_blocking=True)
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
            ra, rb, fa, fb = D_A(diff_aug(rA)), D_B(diff_aug(rB)), D_A(diff_aug(qA)), D_B(diff_aug(qB))
            loss_D = 0.5 * (mse(ra, torch.ones_like(ra) * SM) + mse(fa, torch.zeros_like(fa))) + \
                     0.5 * (mse(rb, torch.ones_like(rb) * SM) + mse(fb, torch.zeros_like(fb)))
        scaler.scale(loss_D).backward(); scaler.step(opt_D); scaler.update()
        sch_G.step(); sch_D.step(); ema_update()
        nan += int(not (torch.isfinite(loss_G) and torch.isfinite(loss_D)))
        sums += [loss_G.item(), loss_D.item(), loss_cyc.item(), loss_id.item()]
    g, d, c, i = sums / CFG["steps_per_epoch"]
    line = f"epoch {epoch} g={g:.4f} d={d:.4f} cycle={c:.4f} id={i:.4f} lr={opt_G.param_groups[0]['lr']:.2e} nan={nan} min={(time.time() - t0) / 60:.1f}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    if epoch > CFG["epochs"] - CFG["score_last"]:
        ck = {"G_AB": E_AB.state_dict(), "G_BA": E_BA.state_dict(), "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
              "epoch": f"93+ft{epoch}_ema", "config": CFG}
        torch.save(ck, OUT / f"ft_ema_{epoch:02d}.pt")
        if on_drive:
            shutil.copy2(OUT / f"ft_ema_{epoch:02d}.pt", DRIVE / f"ft_ema_{epoch:02d}.pt")
            shutil.copy2(LOG, DRIVE / LOG.name)
train_time = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else 0.0
print(f"train_time_sec={train_time:.1f} peak_memory_mb={peak:.1f}", flush=True)
with open(LOG, "a") as f:
    f.write(f"train_time_sec={train_time:.1f} peak_memory_mb={peak:.1f}\n")


# ---------------- score EMA checkpoints (professor method) ----------------
@torch.no_grad()
def translate(G, paths, out, flip):
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.jpg"):
        p.unlink()
    G.eval()
    for k, p in enumerate(paths):
        x = sc.TF(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
        y = G(x)
        if flip:
            y = (y + G(x.flip(-1)).flip(-1)) / 2
        y = y[0].float().cpu().clamp(-1, 1)
        Image.fromarray((((y + 1) * 0.5).permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(out / f"{k:05d}.jpg", quality=95)


def score(ck_path):
    s = torch.load(ck_path, map_location=dev, weights_only=False)
    gab = sc.ResnetGenerator("nearest", ngf=CFG["ngf"]).to(dev); gab.load_state_dict(s["G_AB"])
    gba = sc.ResnetGenerator("nearest", ngf=CFG["ngf"]).to(dev); gba.load_state_dict(s["G_BA"])
    rm, rp = monet_all[:N_EVAL], photo_all[:N_EVAL]
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        translate(gab, rp, t / "b2a", False)
        translate(gba, rm, t / "a2b", False)
        translate(gba, rm, t / "a2b_flip", True)
        fb, mb = sc.ev.calculate_fid_mifid(rm, sc.ev._list_images(t / "b2a")[:N_EVAL], dev)
        fa, ma = sc.ev.calculate_fid_mifid(rp, sc.ev._list_images(t / "a2b")[:N_EVAL], dev)
        ff, mf = sc.ev.calculate_fid_mifid(rp, sc.ev._list_images(t / "a2b_flip")[:N_EVAL], dev)
    sc_plain = ((fa + fb) / 2 + (ma + mb) / 2) / 2
    sc_flip = ((ff + fb) / 2 + (mf + mb) / 2) / 2
    return dict(checkpoint=ck_path.name, fid_b2a=round(fb, 3), mifid_b2a=round(mb, 4), fid_a2b=round(fa, 3), mifid_a2b=round(ma, 4),
                fid_a2b_flip=round(ff, 3), mifid_a2b_flip=round(mf, 4), score_plain=round(sc_plain, 4), score_a2b_flip=round(sc_flip, 4))


rows = []
for ck in sorted(OUT.glob("ft_ema_*.pt")):
    r = score(ck); rows.append(r); print(r, flush=True)
scores_csv = M / "outputs" / "checkpoint_scores_finetune_ema.csv"
scores_csv.parent.mkdir(parents=True, exist_ok=True)
with open(scores_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(sorted(rows, key=lambda r: r["score_a2b_flip"]))
best = min(rows, key=lambda r: r["score_a2b_flip"])
print("\nBEST:", best, "\n(current submission: 50.4667)", flush=True)
if on_drive:
    shutil.copy2(scores_csv, DRIVE / scores_csv.name)


# ---------------- build submission from the best EMA checkpoint, only if it beats 50.4667 ----------------
if best["score_a2b_flip"] < 50.4667 and not SMOKE:
    s = torch.load(OUT / best["checkpoint"], map_location=dev, weights_only=False)
    gab = sc.ResnetGenerator("nearest", ngf=CFG["ngf"]).to(dev); gab.load_state_dict(s["G_AB"])
    gba = sc.ResnetGenerator("nearest", ngf=CFG["ngf"]).to(dev); gba.load_state_dict(s["G_BA"])
    translate(gba, monet_all, M / "outputs" / "pred_A2B", True)    # Monet -> photo, flip averaging
    translate(gab, photo_all, M / "outputs" / "pred_B2A", False)   # photo -> Monet, single pass
    shutil.copy2(OUT / best["checkpoint"], M / "checkpoints" / "best.pt")
    runpy.run_path(str(M / "evaluate_local.py"), run_name="__main__")
    print(open(M / "submission.csv").read(), flush=True)
    os.system(f'cd {LAB} && zip -q -r /content/part3_finetune_ema.zip task3_gan/sneha_singh reproducibility -x "*/data/*" "*/images.zip"')
    if on_drive:
        shutil.copy2("/content/part3_finetune_ema.zip", DRIVE / "part3_finetune_ema.zip")
    print("built /content/part3_finetune_ema.zip", flush=True)
else:
    print("no fine-tuned checkpoint beat 50.4667: keep the epoch-93 submission", flush=True)
