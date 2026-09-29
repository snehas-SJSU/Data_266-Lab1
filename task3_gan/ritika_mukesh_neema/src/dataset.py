"""
Task 3.1(1) - Unpaired image domains.

Loads domain A (e.g. monet_jpg/) and domain B (e.g. photo_jpg/) as two
independent, unpaired image lists -- exactly what CycleGAN needs. Domain B
is sampled with a random offset each __getitem__ so pairs are never aligned
(this is what makes it "unpaired" translation, not paired pix2pix).
"""
import os
import random

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(folder):
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMG_EXTENSIONS)
    )


def make_transform(img_size, train=True):
    ops = [T.Resize((img_size, img_size), Image.BICUBIC)]
    if train:
        ops.append(T.RandomHorizontalFlip())
    ops += [T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]  # -> [-1, 1]
    return T.Compose(ops)


class UnpairedImageDataset(Dataset):
    def __init__(self, dir_a, dir_b, img_size=256, train=True):
        self.files_a = list_images(dir_a)
        self.files_b = list_images(dir_b)
        if len(self.files_a) == 0 or len(self.files_b) == 0:
            raise ValueError(
                f"Found {len(self.files_a)} images in {dir_a} and {len(self.files_b)} in {dir_b} "
                "-- both domains need at least one image."
            )
        self.transform = make_transform(img_size, train)
        self.train = train

    def __len__(self):
        # Epoch length = the SMALLER domain (typically monet_jpg, ~300 images,
        # vs. photo_jpg, ~7000). Using max() here (as some CycleGAN reference
        # implementations do) would make every epoch ~7000 steps, which is
        # impractical on a time-boxed GPU Lab slot. With min(), one epoch = one
        # full pass over the smaller/rarer domain, and photos are still seen
        # via random sampling (see __getitem__) -- just spread across more
        # epochs rather than crammed into one. Total photo coverage over a
        # full training run is unaffected; only wall-clock-per-epoch changes.
        return min(len(self.files_a), len(self.files_b))

    def __getitem__(self, idx):
        path_a = self.files_a[idx % len(self.files_a)]
        # unpaired: random index into B so A/B are never the "same" content
        b_idx = random.randint(0, len(self.files_b) - 1) if self.train else idx % len(self.files_b)
        path_b = self.files_b[b_idx]
        img_a = self.transform(Image.open(path_a).convert("RGB"))
        img_b = self.transform(Image.open(path_b).convert("RGB"))
        return {"A": img_a, "B": img_b, "path_A": path_a, "path_B": path_b}


class SingleDomainDataset(Dataset):
    """Used at inference time to translate every image in one domain, in order."""

    def __init__(self, folder, img_size=256):
        self.files = list_images(folder)
        if len(self.files) == 0:
            raise ValueError(f"No images found in {folder}")
        self.transform = make_transform(img_size, train=False)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = self.transform(Image.open(path).convert("RGB"))
        return {"image": img, "path": path, "filename": os.path.basename(path)}
