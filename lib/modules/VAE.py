import _init_paths
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from lib.BiKLD import BiKLD
from lib.reparameterize import reparameterize


class VAE(nn.Module):
    def __init__(self, indim, latentdim, half=False):
        super(VAE, self).__init__()

        self.half = half
        if self.half is False:
            self.encoder = nn.Sequential(
                nn.Linear(indim, latentdim * 2),
                nn.ELU(inplace=True),
                nn.Linear(latentdim * 2, latentdim * 2),
                nn.ELU(inplace=True),
                nn.Linear(latentdim * 2, latentdim * 2)
            )
            self.mean = nn.Linear(latentdim * 2, latentdim)
            self.logvar = nn.Linear(latentdim * 2, latentdim)
            self.bikld = BiKLD()

        dec_out = indim
        self.decoder = nn.Sequential(
            nn.Linear(latentdim, latentdim * 2),
            nn.ELU(inplace=True),
            nn.Linear(latentdim * 2, latentdim * 2),
            nn.ELU(inplace=True),
            nn.Linear(latentdim * 2, dec_out)
        )

        self.sampler = reparameterize()

    def forward(self, x=None, prior=None):
        prior = [prior[0].view(1, -1), prior[1].view(1, -1)]

        if self.half is False:
            encoding = self.encoder(x)
            mean, logvar = self.mean(encoding), self.logvar(encoding)
            kld = self.bikld([mean, logvar], prior)
            z = self.sampler(mean, logvar)
        else:
            z = self.sampler(prior[0], prior[1])
            kld = 0

        decoding = self.decoder(z)

        return decoding, kld

    def generate(self, prior):
        prior = [prior[0].view(1, -1), prior[1].view(1, -1)]
        z = self.sampler(*prior)

        decoding = self.decoder(z)

        return decoding


'''
#Test case 0
model = VAE(6, 4).cuda()
mean = Variable(torch.zeros(16, 4)).cuda()
var  = Variable(torch.zeros(16, 4)).cuda()
data = Variable(torch.zeros(16, 6)).cuda()
out, kld = model(data, [mean, var])

#Test case 1
model = VAE(6, 4, 10).cuda()
mean = Variable(torch.zeros(16, 4)).cuda()
var  = Variable(torch.zeros(16, 4)).cuda()
data = Variable(torch.zeros(16, 6)).cuda()
condition = Variable(torch.zeros(16, 10)).cuda()
out, kld = model(data, [mean, var], condition)
'''
