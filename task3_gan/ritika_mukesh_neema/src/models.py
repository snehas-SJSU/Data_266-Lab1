"""
Task 3.1(2) - CycleGAN architecture: two generators (G_A2B, G_B2A) and two
discriminators (D_A, D_B). Standard Zhu et al. (2017) design: ResNet-based
generator with instance normalization, 70x70 PatchGAN discriminator.
Implemented from scratch with plain nn.Conv2d/nn.ConvTranspose2d/nn.InstanceNorm2d
building blocks -- no pretrained backbones anywhere in G/D.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """c7s1-64, d128, d256, R256*n_blocks, u128, u64, c7s1-3 (Zhu et al. naming)."""

    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        # downsampling
        ch = ngf
        for _ in range(2):
            model += [
                nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ch * 2),
                nn.ReLU(inplace=True),
            ]
            ch *= 2
        # residual blocks
        for _ in range(n_blocks):
            model += [ResidualBlock(ch)]
        # upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ch // 2),
                nn.ReLU(inplace=True),
            ]
            ch //= 2
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ch, out_ch, kernel_size=7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN: C64-C128-C256-C512, no norm on first layer."""

    def __init__(self, in_ch=3, ndf=64):
        super().__init__()

        def block(cin, cout, stride=2, norm=True):
            layers = [nn.Conv2d(cin, cout, kernel_size=4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        layers += block(in_ch, ndf, norm=False)
        layers += block(ndf, ndf * 2)
        layers += block(ndf * 2, ndf * 4)
        layers += block(ndf * 4, ndf * 8, stride=1)
        layers += [nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # (B, 1, H', W') patch logits


def init_weights(net, gain=0.02):
    def _init(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            nn.init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "InstanceNorm2d" in classname and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(_init)
    return net


def num_params(model):
    return sum(p.numel() for p in model.parameters())
