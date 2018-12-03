import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F


# one more advanced plan: predict an attention map a, then
# render the vw by a*x*y + (1-a)*y
class Describe(nn.Module):
    def __init__(self, hiddim_v=None, hiddim_p=None, op='CAT'):
        super(Describe, self).__init__()
        self.op = op
        self.hiddim_v = hiddim_v
        self.hiddim_p = hiddim_p
        if op == 'CAT' or op == 'CAT_PoE':
            self.net1_mean_vis = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_v * 2, hiddim_v, 3, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
            )

            self.net1_var_vis = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_v * 2, hiddim_v, 3, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
            )

            self.net1_mean_pos = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_p * 2, hiddim_p, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
            )

            self.net1_var_pos = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_p * 2, hiddim_p, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
            )
        elif op == 'DEEP':
            self.net_vis = nn.Sequential(
                weight_norm(nn.Conv2d(4 * hiddim_v, hiddim_v, 3, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_v, 2 * hiddim_v, 3, 1, 1))
            )

            self.net_pos = nn.Sequential(
                weight_norm(nn.Conv2d(4 * hiddim_p, hiddim_p, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_p, 2 * hiddim_p, 1, 1))
            )

        elif op == 'CAT_PROD':
            self.net1_mean_vis = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_v * 2, hiddim_v, 3, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
            )

            self.net1_var_vis = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_v * 2, hiddim_v, 3, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
            )

            self.net2_mean_vis = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
            )

            self.net2_var_vis = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
            )

            self.net1_mean_pos = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_p * 2, hiddim_p, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
            )

            self.net1_var_pos = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_p * 2, hiddim_p, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
            )

            self.net2_mean_pos = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
            )

            self.net2_var_pos = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
            )
        elif op == 'CAT_gPoE':
            self.net1_mean_vis = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_v * 2, hiddim_v, 3, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
            )

            self.net1_var_vis = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_v * 2, hiddim_v, 3, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_v, hiddim_v, 3, 1, 1))
            )

            self.net1_mean_pos = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_p * 2, hiddim_p, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
            )

            self.net1_var_pos = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_p * 2, hiddim_p, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_p, hiddim_p, 1, 1))
            )
            self.gates_v = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_v * 4, hiddim_v * 4, 3, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_v * 4, hiddim_v * 4, 3, 1, 1))
            )
            self.gates_p = nn.Sequential(
                weight_norm(nn.Conv2d(hiddim_p * 4, hiddim_p * 4, 3, 1, 1)),
                nn.ELU(inplace=True),
                weight_norm(nn.Conv2d(hiddim_p * 4, hiddim_p * 4, 3, 1, 1))
            )

    def forward(self, x, y, mode, lognormal=False):  # -> x describe y
        if mode == 'vis':
            if self.op == 'CAT_PROD':
                x_mean = self.net1_mean_vis(torch.cat([x[0], y[0]], dim=1))
                x_var = self.net1_var_vis(torch.cat([x[1], y[1]], dim=1))

                if lognormal == True:
                    x_mean = torch.exp(x_mean)

                y_mean = self.net2_mean_vis(x_mean * y[0])
                y_var = self.net2_var_vis(x_var * y[1])
            elif self.op == 'CAT_PoE':
                # logvar = -log(exp(-logvar1) + exp(-logvar2))
                # mu     = exp(logvar) * (exp(-logvar1) * mu1 + exp(-logvar2) * mu2)
                x_mean = self.net1_mean_vis(torch.cat([x[0], y[0]], dim=1))
                x_var = self.net1_var_vis(torch.cat([x[1], y[1]], dim=1))
                mlogvar1 = -x_var
                mlogvar2 = -y[1]
                mu1 = x_mean
                mu2 = y[0]

                y_var = -torch.log(torch.exp(mlogvar1) + torch.exp(mlogvar2))
                y_mean = torch.exp(y_var) * (torch.exp(mlogvar1) * mu1 + torch.exp(mlogvar2) * mu2)
            elif self.op == 'CAT_gPoE':
                # logvar = -log(exp(-logvar1) + exp(-logvar2))
                # mu     = exp(logvar) * (exp(-logvar1) * mu1 + exp(-logvar2) * mu2)
                x_mean = self.net1_mean_vis(torch.cat([x[0], y[0]], dim=1))
                x_var = self.net1_var_vis(torch.cat([x[1], y[1]], dim=1))

                # gates
                gates = torch.sigmoid(self.gates_v(torch.cat([x_mean, x_var, y[0], y[1]], dim=1)))
                x1_mu_g = gates[:, :self.hiddim_v, :, :]
                x1_var_g = gates[:, self.hiddim_v:2 * self.hiddim_v, :, :]
                x2_mu_g = gates[:, 2 * self.hiddim_v:3 * self.hiddim_v, :, :]
                x2_var_g = gates[:, 3 * self.hiddim_v:4 * self.hiddim_v, :, :]

                x_mean = x1_mu_g * x_mean
                x_var = torch.log(x1_var_g + 1e-5) + x_var
                y[0] = x2_mu_g * y[0]
                y[1] = torch.log(x2_var_g + 1e-5) + y[1]

                mlogvar1 = -x_var
                mlogvar2 = -y[1]
                mu1 = x_mean
                mu2 = y[0]

                y_var = -torch.log(torch.exp(mlogvar1) + torch.exp(mlogvar2))
                y_mean = torch.exp(y_var) * (torch.exp(mlogvar1) * mu1 + torch.exp(mlogvar2) * mu2)
            elif self.op == 'CAT':
                y_mean = self.net1_mean_vis(torch.cat([x[0], y[0]], dim=1))
                y_var = self.net1_var_vis(torch.cat([x[1], y[1]], dim=1))
            elif self.op == 'PROD':
                y_mean = x[0] * y[0]
                y_var = x[1] * y[1]
            elif self.op == 'DEEP':
                gaussian_out = self.net_vis(torch.cat([x[0], x[1], y[0], y[1]], dim=1))
                y_mean = gaussian_out[:, :self.hiddim_v, :, :]
                y_var = gaussian_out[:, self.hiddim_v:, :, :]
            else:
                raise ValueError('invalid operator name {} for Describe module'.format(self.op))

        elif mode == 'pos':
            if self.op == 'CAT_PROD':
                x_mean = self.net1_mean_pos(torch.cat([x[0], y[0]], dim=1))
                x_var = self.net1_var_pos(torch.cat([x[1], y[1]], dim=1))

                y_mean = self.net2_mean_pos(x_mean * y[0])
                y_var = self.net2_var_pos(x_var * y[1])
            elif self.op == 'CAT_PoE':
                # logvar = -log(exp(-logvar1) + exp(-logvar2))
                # mu     = exp(logvar) * (exp(-logvar1) * mu1 + exp(-logvar2) * mu2)
                x_mean = self.net1_mean_pos(torch.cat([x[0], y[0]], dim=1))
                x_var = self.net1_var_pos(torch.cat([x[1], y[1]], dim=1))

                mlogvar1 = -x_var
                mlogvar2 = -y[1]
                mu1 = x_mean
                mu2 = y[0]

                y_var = -torch.log(torch.exp(mlogvar1) + torch.exp(mlogvar2))
                y_mean = torch.exp(y_var) * (torch.exp(mlogvar1) * mu1 + torch.exp(mlogvar2) * mu2)
            elif self.op == 'CAT_gPoE':
                # logvar = -log(exp(-logvar1) + exp(-logvar2))
                # mu     = exp(logvar) * (exp(-logvar1) * mu1 + exp(-logvar2) * mu2)
                x_mean = self.net1_mean_pos(torch.cat([x[0], y[0]], dim=1))
                x_var = self.net1_var_pos(torch.cat([x[1], y[1]], dim=1))

                # gates
                gates = torch.sigmoid(self.gates_p(torch.cat([x_mean, x_var, y[0], y[1]], dim=1)))
                x1_mu_g = gates[:, :self.hiddim_p, :, :]
                x1_var_g = gates[:, self.hiddim_p:2 * self.hiddim_p, :, :]
                x2_mu_g = gates[:, 2 * self.hiddim_p:3 * self.hiddim_p, :, :]
                x2_var_g = gates[:, 3 * self.hiddim_p:4 * self.hiddim_p, :, :]

                x_mean = x1_mu_g * x_mean
                x_var = torch.log(x1_var_g + 1e-5) + x_var
                y[0] = x2_mu_g * y[0]
                y[1] = torch.log(x2_var_g + 1e-5) + y[1]

                mlogvar1 = -x_var
                mlogvar2 = -y[1]
                mu1 = x_mean
                mu2 = y[0]

                y_var = -torch.log(torch.exp(mlogvar1) + torch.exp(mlogvar2))
                y_mean = torch.exp(y_var) * (torch.exp(mlogvar1) * mu1 + torch.exp(mlogvar2) * mu2)
            elif self.op == 'CAT':
                y_mean = self.net1_mean_pos(torch.cat([x[0], y[0]], dim=1))
                y_var = self.net1_var_pos(torch.cat([x[1], y[1]], dim=1))
            elif self.op == 'PROD':
                y_mean = x[0] * y[0]
                y_var = x[1] * y[1]
            elif self.op == 'DEEP':
                gaussian_out = self.net_pos(torch.cat([x[0], x[1], y[0], y[1]], dim=1))
                y_mean = gaussian_out[:, :self.hiddim_p, :, :]
                y_var = gaussian_out[:, self.hiddim_p:, :, :]
            else:
                raise ValueError('invalid operator name {} for Describe module'.format(self.op))

        else:
            raise ValueError('invalid mode {}'.format(mode))

        return [y_mean, y_var]
