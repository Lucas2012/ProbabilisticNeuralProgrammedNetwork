import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class Transform(nn.Module):
    def __init__(self, matrix='default'):
        super(Transform, self).__init__()

    def forward(self, x, hw, variance=False):
        if variance:
            x = torch.exp(x / 2.0)
        size = torch.Size([x.size(0), x.size(1), int(hw[0]), int(hw[1])])

        # grid generation
        theta = np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.float32)
        theta = Variable(torch.from_numpy(theta), requires_grad=False).cuda()
        theta = theta.expand(x.size(0), theta.size(1), theta.size(2))
        gridout = F.affine_grid(theta, size)

        # bilinear sampling
        out = F.grid_sample(x, gridout, mode='bilinear')
        if variance:
            out = torch.log(out) * 2.0
        return out
