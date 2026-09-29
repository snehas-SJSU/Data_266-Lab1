"""Shared training utilities: image replay buffer, LR schedule, denorm helper."""
import random
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
