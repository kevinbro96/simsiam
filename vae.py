from __future__ import print_function
import abc
import os
import math
import pdb
import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable

import torch.nn.functional as F
from torch.autograd import Function, Variable
import functools


class MLP(nn.Module):
    def __init__(
        self,
        in_features=2048,
        hidden_layers=[],
        num_classes=10,
        activation="relu",
        bn=True,
        dropout=0.0,
    ):
        super().__init__()
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]

        assert len(hidden_layers) > 0
        self.out_features = hidden_layers[-1]

        mlp = []

        if activation == "relu":
            act_fn = functools.partial(nn.ReLU, inplace=True)
        elif activation == "leaky_relu":
            act_fn = functools.partial(nn.LeakyReLU, inplace=True)
        else:
            raise NotImplementedError

        for hidden_dim in hidden_layers:
            mlp += [nn.Linear(in_features, hidden_dim)]
            if bn:
                mlp += [nn.BatchNorm1d(hidden_dim)]
            mlp += [act_fn()]
            if dropout > 0:
                mlp += [nn.Dropout(dropout)]
            in_features = hidden_dim
        mlp += [nn.Linear(self.out_features, num_classes)]
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return


class CVAE_imagenet_withbn(AbstractAutoEncoder):
    def __init__(self, d, z,  **kwargs):
        super(CVAE_imagenet_withbn, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 16, d // 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 8, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 4, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d)
        )

        self.decoder = nn.Sequential(
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 4, d // 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 8),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 8, d // 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 16),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 16, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.bn = nn.BatchNorm2d(3)
        self.f = 7
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        mu = self.fc11(h1)
        logvar = self.fc12(h1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = self.fc21(z)
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return self.bn(torch.tanh(h3))

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        gx = self.decode(z)
            
        return z, gx


class CVAE_imagenet_withbn_v2(AbstractAutoEncoder):
    def __init__(self, d, z,  **kwargs):
        super(CVAE_imagenet_withbn_v2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 2, kernel_size=4, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=4, padding=0, bias=False),
        )
        self.bn = nn.BatchNorm2d(3)
        self.f = 14
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        mu = self.fc11(h1)
        logvar = self.fc12(h1)
        z = self.reparameterizae(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = self.fc21(z)
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return self.bn(torch.tanh(h3))

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        gx = self.decode(z)
        return z, gx, mu, logvar
