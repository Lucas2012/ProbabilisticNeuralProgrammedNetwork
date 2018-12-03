import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F

class DistributionRender(nn.Module):
  def __init__(self, hiddim):
    super(DistributionRender, self).__init__()
    self.render_mean = nn.Sequential(
                      nn.Conv2d(hiddim, hiddim, 3, 1, 1),
                      nn.ELU(inplace=True),
                      nn.Conv2d(hiddim, hiddim, 3, 1, 1),
                      nn.ELU(inplace=True),
                      nn.Conv2d(hiddim, hiddim, 3, 1, 1),
                  )

    self.render_var  = nn.Sequential(
                      nn.Conv2d(hiddim, hiddim, 3, 1, 1),
                      nn.ELU(inplace=True),
                      nn.Conv2d(hiddim, hiddim, 3, 1, 1),
                      nn.ELU(inplace=True),
                      nn.Conv2d(hiddim, hiddim, 3, 1, 1),
                  )


  def forward(self, x):
    # x = [mean, var]
    return self.render_mean(x[0]), self.render_var(x[1])
