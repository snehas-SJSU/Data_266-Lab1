"""Full Monet→photo (B2A) inference from best.pt. Does not touch pred_A2B."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
MEMBER = Path(__file__).resolve().parent
MONET_DIR = ROOT / "task3_gan" / "data" / "monet_jpg"
PRED_B2A = MEMBER / "outputs" / "pred_B2A"
CKPT = MEMBER / "checkpoints" / "best.pt"
IMG = 256


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf, affine=False, track_running_stats=False),
            nn.ReLU(True),
        ]
        n = ngf
        for _ in range(2):
            model += [
                nn.Conv2d(n, n * 2, 3, 2, 1, bias=False),
                nn.InstanceNorm2d(n * 2, affine=False, track_running_stats=False),
                nn.ReLU(True),
            ]
            n *= 2
        for _ in range(n_blocks):
            model += [ResnetBlock(n)]
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(n, n // 2, 3, 2, 1, output_padding=1, bias=False),
                nn.InstanceNorm2d(n // 2, affine=False, track_running_stats=False),
                nn.ReLU(True),
            ]
            n //= 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def list_images(folder):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in exts])


def tensor_to_pil(t):
    t = t.detach().cpu().clamp(-1, 1)
    t = (t + 1) * 0.5
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device:", device)

    paths = list_images(MONET_DIR)
    if not paths:
        raise FileNotFoundError(f"No Monet JPGs in {MONET_DIR}")
    if not CKPT.exists():
        raise FileNotFoundError(f"Missing {CKPT}")

    tf = transforms.Compose(
        [
            transforms.Resize((IMG, IMG), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    G_BA = ResnetGenerator(ngf=64, n_blocks=9).to(device)
    state = torch.load(CKPT, map_location=device, weights_only=True)
    G_BA.load_state_dict(state["G_BA"])
    G_BA.eval()

    PRED_B2A.mkdir(parents=True, exist_ok=True)
    for p in PRED_B2A.glob("*"):
        if p.is_file() and p.name != ".gitkeep":
            p.unlink()

    print(f"writing {len(paths)} B2A preds → {PRED_B2A}")
    with torch.no_grad():
        for i, path in enumerate(paths):
            x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            y = G_BA(x)[0]
            tensor_to_pil(y).save(PRED_B2A / f"{i:05d}.jpg", quality=95)
            if (i + 1) % 50 == 0 or i + 1 == len(paths):
                print(f"  {i + 1}/{len(paths)}", flush=True)
    print("done. pred_B2A count:", len(list_images(PRED_B2A)))


if __name__ == "__main__":
    main()
