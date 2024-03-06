from torch import nn as nn
import torch
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

seq = nn.Sequential

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass

def conv2d(ch_in, ch_out, kz, s=1, p=0):
    return spectral_norm(nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=kz, stride=s, padding=p))


class LeNet(nn.Module):
    def __init__(self, num_class) -> None:
        super().__init__()

        self.block = seq(
            conv2d(ch_in=1, ch_out=6, kz=5, s=1, p=0), nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            conv2d(ch_in=6, ch_out=16, kz=5, s=1, p=0), nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=400,out_features=120), nn.Tanh(),
            nn.Linear(in_features=120,out_features=84), nn.Tanh(),
            nn.Linear(in_features=84,out_features=num_class), nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)