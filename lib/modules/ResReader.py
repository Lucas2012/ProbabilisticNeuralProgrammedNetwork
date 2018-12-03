import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from lib.ResidualModule import ResidualModule


class Reader(nn.Module):
    def __init__(self, indim, hiddim, outdim, ds_times, normalize, nlayers=4):
        super(Reader, self).__init__()

        self.ds_times = ds_times

        if normalize == 'gate':
          ifgate = True
        else:
          ifgate = False

        self.encoder = ResidualModule(modeltype='encoder', indim=indim, hiddim=hiddim, outdim=outdim,
                                      nres=self.ds_times, nlayers=nlayers, ifgate=ifgate, normalize=normalize)

    def forward(self, x):
        out = self.encoder(x)

        return out
