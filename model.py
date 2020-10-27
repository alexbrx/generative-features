import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, depth_concat=True):
        super(Generator, self).__init__()

        self.depth_concat = depth_concat
        layers = []
        if self.depth_concat:
            layers.append(
                nn.Conv2d(
                    3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False
                )
            )
        else:
            layers.append(
                nn.Conv2d(4, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
            )
        layers.append(
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        )
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(
                nn.Conv2d(
                    curr_dim,
                    curr_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True)
            )
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(
                nn.ConvTranspose2d(
                    curr_dim,
                    curr_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True)
            )
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        )
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        if self.depth_concat:
            c = c.view(c.size(0), c.size(1), 1, 1)
            c = c.repeat(1, 1, x.size(2), x.size(3))
        else:
            if (
                c.size(1) == 2
            ):  # labels are assumed to have length 2 or 64 and images are 112x112
                c = (
                    c.view(-1)
                    .repeat_interleave(56)
                    .view(c.size(0), 1, 1, -1)
                    .repeat(1, 1, 112, 1)
                )
            elif c.size(1) == 64:
                c = c.view(c.size(0), 1, 8, 8)
                c = c.repeat(1, 1, 14, 14)

        x = torch.cat([x, c], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(
        self,
        image_size=128,
        conv_dim=64,
        c_dim=5,
        repeat_num=6,
        affectnet_emo_descr=None,
        d_loss_cls_type=None,
    ):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(
            curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

        if affectnet_emo_descr == "64d_cls" and (d_loss_cls_type in ["both", "pred"]):
            self.fc = torch.nn.Linear(c_dim, 7)
        elif affectnet_emo_descr == "64d_reg" and (d_loss_cls_type in ["both", "pred"]):
            self.fc = torch.nn.Linear(c_dim, 2)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
