import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.utils import weight_norm


class ResidualModule(nn.Module):
    def __init__(self, modeltype, indim, hiddim, outdim, nlayers, nres, ifgate=False, nonlinear='elu', normalize='instance_norm'):
        super(ResidualModule, self).__init__()
        if ifgate:
          print('Using gated version.')
        if modeltype == 'encoder':
            self.model = self.encoder(indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize=normalize)
        elif modeltype == 'decoder':
            self.model = self.decoder(indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize=normalize)
        elif modeltype == 'plain':
            self.model = self.plain(indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize=normalize)
        else:
            raise ('Uknown model type.')

    def encoder(self, indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize):
        layers = []
        layers.append(ResidualBlock(None, nonlinear, ifgate, indim, hiddim, normalize))

        for i in range(0, nres):
            for j in range(0, nlayers):
                layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, hiddim, normalize))
            layers.append(ResidualBlock('down', nonlinear, ifgate, hiddim, hiddim, normalize))

        layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, outdim, normalize))

        return nn.Sequential(*layers)

    def decoder(self, indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize):
        layers = []
        layers.append(ResidualBlock(None, nonlinear, ifgate, indim, hiddim, normalize))

        for i in range(0, nres):
            for j in range(0, nlayers):
                layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, hiddim, normalize))
            layers.append(ResidualBlock('up', nonlinear, ifgate, hiddim, hiddim, normalize))

        layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, outdim, normalize))

        return nn.Sequential(*layers)

    def plain(self, indim, hiddim, outdim, nlayers, nres, ifgate, nonlinear, normalize):
        layers = []
        layers.append(ResidualBlock(None, nonlinear, ifgate, indim, hiddim, normalize))

        for i in range(0, nres):
            for j in range(0, nlayers):
                layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, hiddim, normalize))

        layers.append(ResidualBlock(None, nonlinear, ifgate, hiddim, outdim, normalize))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, resample, nonlinear, ifgate, indim, outdim, normalize):
        super(ResidualBlock, self).__init__()

        self.ifgate = ifgate
        self.indim = indim
        self.outdim = outdim
        self.resample = resample

        if resample == 'down':
            convtype = 'sconv_d'
        elif resample == 'up':
            convtype = 'upconv'
        elif resample == None:
            convtype = 'sconv'

        self.shortflag = False
        if not (indim == outdim and resample == None):
            self.shortcut = self.conv(convtype, indim, outdim)
            self.shortflag = True

        if ifgate:
            self.conv1 = nn.Conv2d(indim, outdim, 3, 1, 1)
            self.conv2 = nn.Conv2d(indim, outdim, 3, 1, 1)
            self.c = nn.Sigmoid()
            self.g = nn.Tanh()
            self.conv3 = self.conv(convtype, outdim, outdim)
            self.act = self.nonlinear(nonlinear)
        elif normalize == 'batch_norm':
            self.resblock = nn.Sequential(
                self.conv('sconv', indim, outdim),
                nn.BatchNorm2d(outdim),
                self.nonlinear(nonlinear),
                self.conv(convtype, outdim, outdim),
                nn.BatchNorm2d(outdim),
                self.nonlinear(nonlinear)
            )
        elif normalize == 'instance_norm':
            self.resblock = nn.Sequential(
                self.conv('sconv', indim, outdim),
                nn.InstanceNorm2d(outdim),
                self.nonlinear(nonlinear),
                self.conv(convtype, outdim, outdim),
                nn.InstanceNorm2d(outdim),
                self.nonlinear(nonlinear)
            )
        elif normalize == 'no_norm':
            self.resblock = nn.Sequential(
                self.conv('sconv', indim, outdim),
                self.nonlinear(nonlinear),
                self.conv(convtype, outdim, outdim),
                self.nonlinear(nonlinear)
            )
        elif normalize == 'weight_norm':
            self.resblock = nn.Sequential(
                self.conv('sconv', indim, outdim, 'weight_norm'),
                self.nonlinear(nonlinear),
                self.conv(convtype, outdim, outdim, 'weight_norm'),
                self.nonlinear(nonlinear)
            )

    def conv(self, name, indim, outdim, normalize=None):
        if name == 'sconv_d':
            if normalize == 'weight_norm':
              return weight_norm(nn.Conv2d(indim, outdim, 4, 2, 1))
            else:
              return nn.Conv2d(indim, outdim, 4, 2, 1)
        elif name == 'sconv':
            if normalize == 'weight_norm':
              return weight_norm(nn.Conv2d(indim, outdim, 3, 1, 1))
            else:
              return nn.Conv2d(indim, outdim, 3, 1, 1)
        elif name == 'upconv':
            if normalize == 'weight_norm':
              return weight_norm(nn.ConvTranspose2d(indim, outdim, 4, 2, 1))
            else:
              return nn.ConvTranspose2d(indim, outdim, 4, 2, 1)
        else:
            raise ("Unknown convolution type")

    def nonlinear(self, name):
        if name == 'elu':
            return nn.ELU(1, True)
        elif name == 'relu':
            return nn.ReLU(True)

    def forward(self, x):
        if self.ifgate:
            conv1 = self.conv1(x)
            conv2 = self.conv2(x)
            c = self.c(conv1)
            g = self.g(conv2)
            gated = c * g
            conv3 = self.conv3(gated)
            res = self.act(conv3)
            if not (self.indim == self.outdim and self.resample == None):
                out = self.shortcut(x) + res
            else:
                out = x + res
        else:
            if self.shortflag:
                out = self.shortcut(x) + self.resblock(x)
            else:
                out = x + self.resblock(x)

        return out
