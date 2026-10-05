"""
Inference helpers shared by train.py (checkpoint scoring), generate.py and
full_metrics.py: build generators from a checkpoint's own config, translate
with optional test-time view averaging, and save JPEGs at high quality.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models import ResnetGenerator

# Test-time view averaging for Monet -> photo (G_A2B). Each view is (hflip, dy, dx):
# the input is flipped / shifted (reflect padded), translated, then the output is
# shifted / flipped back and all views are averaged. Averaging cancels part of the
# generator's view-dependent noise. Measured on this task it lowers Monet->photo FID,
# while for photo->Monet averaging blurs the brush texture, so that direction stays single pass.
VIEW_SETS = {
    1: [(False, 0, 0)],
    2: [(False, 0, 0), (True, 0, 0)],
    6: [(False, 0, 0), (True, 0, 0), (False, 4, 0), (False, 0, 4), (True, -4, 0), (True, 0, -4)],
}


def build_generators(device, ngf=64, n_blocks=9, upsample="convtranspose"):
    G_A2B = ResnetGenerator(ngf=ngf, n_blocks=n_blocks, upsample=upsample).to(device)  # monet -> photo
    G_B2A = ResnetGenerator(ngf=ngf, n_blocks=n_blocks, upsample=upsample).to(device)  # photo -> monet
    return G_A2B, G_B2A


def load_generators(ckpt_path, device, ngf=None, n_blocks=None, upsample=None):
    """Rebuild both generators from a checkpoint. Architecture settings come from the
    checkpoint's saved config unless overridden; old checkpoints (no ngf / upsample in
    their config) fall back to the original 64-filter transposed-conv decoder."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    G_A2B, G_B2A = build_generators(
        device,
        ngf=ngf or cfg.get("ngf", 64),
        n_blocks=n_blocks or cfg.get("n_blocks", 9),
        upsample=upsample or cfg.get("upsample", "convtranspose"),
    )
    G_A2B.load_state_dict(ckpt["G_A2B"])
    G_B2A.load_state_dict(ckpt["G_B2A"])
    return G_A2B.eval(), G_B2A.eval(), cfg, ckpt


def _shift(x, dy, dx):
    p = max(abs(dy), abs(dx))
    if p == 0:
        return x
    h, w = x.shape[-2:]
    return F.pad(x, (p, p, p, p), mode="reflect")[..., p + dy:p + dy + h, p + dx:p + dx + w]


@torch.no_grad()
def translate(G, x, n_views=1):
    """G(x) averaged over VIEW_SETS[n_views]; x is a batch in [-1, 1]."""
    out = 0
    views = VIEW_SETS[n_views]
    for flip, dy, dx in views:
        inp = _shift(x.flip(-1) if flip else x, dy, dx)
        y = _shift(G(inp), -dy, -dx)
        out = out + (y.flip(-1) if flip else y)
    return out / len(views)


def save_jpg(t, path, quality=95):
    """Save one [-1, 1] CHW tensor as a JPEG. torchvision's save_image uses PIL's default
    JPEG quality (75); 95 keeps compression artifacts out of the evaluated images."""
    arr = ((t.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(np.ascontiguousarray(arr)).save(path, quality=quality)
