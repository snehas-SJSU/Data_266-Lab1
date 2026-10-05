"""Shared training utilities: image replay buffer, LR schedule, denorm helper,
EMA of generator weights, GPU thermal guard."""
import copy
import random
import subprocess
import time

import torch


class ImagePool:
    """
    Replay buffer of previously generated fakes (Shrivastava et al. 2017 / the
    CycleGAN paper section 4): stabilizes discriminator training by showing it
    a history of generated images instead of only the very latest batch.
    """

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        out = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img)
                out.append(img)
            elif random.random() > 0.5:
                idx = random.randint(0, self.pool_size - 1)
                out.append(self.images[idx].clone())
                self.images[idx] = img
            else:
                out.append(img)
        return torch.cat(out, dim=0)


def lambda_lr(epoch, n_epochs, n_epochs_decay):
    """1.0 for the first n_epochs, then linearly decays to 0 over n_epochs_decay."""
    if epoch < n_epochs:
        return 1.0
    return max(0.0, 1.0 - (epoch - n_epochs) / float(n_epochs_decay))


def denorm(x):
    """[-1, 1] -> [0, 1] for saving/viewing images."""
    return (x * 0.5 + 0.5).clamp(0, 1)


class EMA:
    """
    Exponential moving average of a model's weights (Polyak averaging). After
    every optimizer step: shadow = decay * shadow + (1 - decay) * weights.
    With decay 0.999 the shadow is roughly an average over the last ~1,000
    steps, which smooths out step-to-step GAN noise; the shadow generators are
    scored alongside the raw ones and whichever is better is kept.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ps, p in zip(self.shadow.parameters(), model.parameters()):
            ps.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)


def gpu_temperature():
    """Current GPU temperature in C via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return int(out[0])
    except Exception:
        return None


def cool_down(max_temp, resume_temp, log=print):
    """Thermal guard for laptop runs: if the GPU is at/above max_temp, sleep until it
    cools to resume_temp. Pausing between steps does not change the training math."""
    if not max_temp:
        return
    t = gpu_temperature()
    if t is None or t < max_temp:
        return
    log(f"  [thermal] GPU at {t} C, pausing until {resume_temp} C ...")
    while t is not None and t > resume_temp:
        time.sleep(20)
        t = gpu_temperature()
    log(f"  [thermal] cooled to {t} C, resuming")
