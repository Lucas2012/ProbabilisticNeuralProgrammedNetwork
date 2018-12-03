import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class ConceptMapper(nn.Module):
  def __init__(self, CHW, vocab_size):
    super(ConceptMapper, self).__init__()
    C, H, W = CHW[0], CHW[1], CHW[2]
    self.mean_dictionary = nn.Linear(vocab_size, C*H*W, bias=False)
    self.std_dictionary  = nn.Linear(vocab_size, C*H*W, bias=False)
    self.C, self.H, self.W = C, H, W

  def forward(self, x):
    word_mean = self.mean_dictionary(x)
    word_std  = self.std_dictionary(x)
    if self.H == 1 and self.W == 1:
      return [word_mean.view(-1, self.C, 1, 1), \
              word_std.view(-1, self.C, 1, 1)]
    else:
      return [word_mean.view(-1, self.C, self.H, self.W), \
              word_std.view(-1, self.C, self.H, self.W)]

