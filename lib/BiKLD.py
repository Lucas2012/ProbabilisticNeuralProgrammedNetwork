import torch as t
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class BiKLD(nn.Module):
    def __init__(self):
        super(BiKLD, self).__init__()

    def forward(self, q, p):
        q_mu, q_var = q[0], t.exp(q[1])
        p_mu, p_var = p[0], t.exp(p[1])

        kld = q_var / p_var - 1
        kld += (p_mu - q_mu).pow(2) / p_var
        kld += p[1] - q[1]
        kld = kld.sum() / 2

        return kld
