import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np


class BiKLD(nn.Module):
    def __init__(self, lambda_t, k):
        super(BiKLD, self).__init__()
        print('Using the modified KL Divergence analytic solution, thresholded by lambda:', lambda_t)
        self.lambda_t = lambda_t
        self.k = k

    def sample(self, mu, logvar, k):
        # input:  [B, C, H, W], [B, C, H, W]
        # output: [B, K, C, H, W]
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def gaussian_diag_logps(self, mu, logvar, z):
        logps = logvar + ((z - mu) ** 2) / (logvar.exp())
        logps = logps.add(np.log(2 * np.pi))
        logps = logps.mul(-0.5)
        return logps

    def expand_dis(self, p, k):
        mu, logvar = p[0], p[1]
        logvar = logvar.unsqueeze(1).expand(logvar.size(0), k, logvar.size(1), logvar.size(2), logvar.size(3))
        mu = mu.unsqueeze(1).expand(mu.size(0), k, mu.size(1), mu.size(2), mu.size(3))
        return mu, logvar

    def forward(self, q, p):
        # input:  [B, C, H, W], [B, C, H, W]
        # output: [1]
        # expand
        q_mu, q_var = q[0], torch.exp(q[1])
        p_mu, p_var = p[0], torch.exp(p[1])

        kld = q_var / p_var - 1
        kld += (p_mu - q_mu).pow(2) / p_var
        kld += p[1] - q[1]
        kld = kld.sum(dim=3).sum(dim=2).mean(0) / 2

        lambda_tensor = Variable(self.lambda_t * torch.ones(kld.size())).cuda()
        kld = torch.max(kld, lambda_tensor)
        kld = kld.sum() * p[0].size(0)

        return kld


if __name__ == '__main__':
    # test code
    q_mu = Variable(torch.zeros(4, 32, 16, 16)).cuda()
    q_logvar = Variable(torch.zeros(4, 32, 16, 16)).cuda()
    p_mu = Variable(torch.zeros(4, 32, 16, 16)).cuda()
    p_logvar = Variable(torch.zeros(4, 32, 16, 16)).cuda()

    kldloss = BiKLD(lambda_t=0.01, k=10)

    kld = kldloss([q_mu, q_logvar], [p_mu, p_logvar])
