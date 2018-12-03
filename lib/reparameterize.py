import torch
import torch.nn as nn
from torch.autograd import Variable


class reparameterize(nn.Module):
    def __init__(self):
        super(reparameterize, self).__init__()

    def forward(self, mu, logvar, sample_num=1, phase='training'):
        if phase == 'training':
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            raise ValueError('Wrong phase. Always assume training phase.')
        # elif phase == 'test':
        #  return mu
        # elif phase == 'generation':
        #  eps = Variable(logvar.data.new(logvar.size()).normal_())
        #  return eps
