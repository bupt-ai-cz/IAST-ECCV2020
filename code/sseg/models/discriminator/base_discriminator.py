import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.snconv import SNConv2d
from ..ops.spp import PPM

from ..registry import DISCRIMINATOR

class FCDiscriminator(nn.Module):
    def __init__(self, in_channels, ndf = 64, use_pool=False, pool_bin=None):
        super(FCDiscriminator, self).__init__()
        if pool_bin:
            self.pool_bin = pool_bin
        else:
            self.pool_bin = (2, 4, 8, 16)
        self.use_pool = use_pool
        if self.use_pool:
            self.conv1 = nn.Conv2d(in_channels*len(self.pool_bin), ndf, kernel_size=3, stride=2, padding=1)
            self.pools = Pools(in_channels, in_channels, self.pool_bin)
        else:
            self.conv1 = nn.Conv2d(in_channels, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, img):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[-1]
        if self.use_pool:
            img = self.pools(img)
        x = self.conv1(img)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

@DISCRIMINATOR.register("Origin-Predictor")
def base_discriminator(channels):
    return FCDiscriminator(channels[-1])
