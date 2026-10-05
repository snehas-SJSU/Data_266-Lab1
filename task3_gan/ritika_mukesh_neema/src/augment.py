"""
Differentiable augmentation for the discriminators (DiffAugment, Zhao et al. 2020,
"Differentiable Augmentation for Data-Efficient GAN Training").

With only 300 Monet paintings, D_A can memorise the real set and stop giving the
generator useful gradients. DiffAugment applies the same family of random
transforms to EVERY image a discriminator sees -- real images, pooled fakes, and
the fakes in the generator's adversarial loss -- so D cannot memorise exact
pixels, while the transforms stay differentiable so gradients still reach G.

Policies (comma separated, e.g. "translation,cutout"):
  color        random brightness / saturation / contrast
  translation  random shift of up to 1/8 of the image, zero padded
  cutout       one random square (half the image side) set to zero
Images are in [-1, 1].
"""
import torch
import torch.nn.functional as F


def _brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) - 0.5)


def _saturation(x):
    gray = x.mean(dim=1, keepdim=True)
    return gray + (x - gray) * (2.0 * torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype))


def _contrast(x):
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return mean + (x - mean) * (0.5 + torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype))


def _translation(x, ratio=0.125):
    n, _, h, w = x.shape
    ph, pw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    padded = F.pad(x, (pw, pw, ph, ph))
    dy = torch.randint(0, 2 * ph + 1, (n,)).tolist()
    dx = torch.randint(0, 2 * pw + 1, (n,)).tolist()
    return torch.cat([padded[i:i + 1, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w] for i in range(n)], dim=0)


def _cutout(x, ratio=0.5):
    n, _, h, w = x.shape
    ch, cw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    mask = torch.ones(n, 1, h, w, device=x.device, dtype=x.dtype)
    cy = torch.randint(0, h, (n,)).tolist()
    cx = torch.randint(0, w, (n,)).tolist()
    for i in range(n):
        y0, x0 = max(cy[i] - ch // 2, 0), max(cx[i] - cw // 2, 0)
        mask[i, :, y0:min(cy[i] + ch // 2, h), x0:min(cx[i] + cw // 2, w)] = 0
    return x * mask


POLICIES = {
    "color": [_brightness, _saturation, _contrast],
    "translation": [_translation],
    "cutout": [_cutout],
}


def diff_augment(x, policy=""):
    """Apply the policy's transforms in order; an empty policy returns x unchanged."""
    for name in [p.strip() for p in policy.split(",") if p.strip()]:
        for fn in POLICIES[name]:
            x = fn(x)
    return x
