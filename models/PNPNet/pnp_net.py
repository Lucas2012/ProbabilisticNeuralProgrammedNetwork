'''
  *
  PNP-Net: flexibly takes a tree-structure program and
           assemble modules for image generative modeling
  *

  It contains:

  -- primitive visual elements

  -- unit modules

  -- tree recursive function

  -- forward(), generate()

  -- utility functions: clean_tree(), etc.
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math

from lib.reparameterize import reparameterize
from lib.modules.VAE import VAE
from lib.modules.ResReader import Reader
from lib.modules.ResWriter import Writer
from lib.modules.ConceptMapper import ConceptMapper
from lib.modules.Combine import Combine
from lib.modules.Describe import Describe
from lib.modules.Transform import Transform
from lib.modules.DistributionRender import DistributionRender


class PNPNet(nn.Module):
    def __init__(self, hiddim=160, latentdim=12,
                 word_size=[-1, 16, 16], pos_size=[4, 1, 1], nres=4, nlayers=1,
                 nonlinear='elu', dictionary=None, op=['PROD', 'CAT'],
                 lmap_size=0, downsample=2, gpu_ids=None,
                 multigpu_full=False, lambdakl=-1, bg_bias=False, normalize='instance_norm',
                 loss=None, debug_mode=True):
        super(PNPNet, self).__init__()
        ## basic settings
        word_size[0] = latentdim
        self.word_size = word_size
        self.latentdim = latentdim
        self.hiddim = hiddim
        self.downsample = downsample  # -> downsample times
        self.ds = 2 ** self.downsample  # -> downsample ratio
        self.nres = nres
        self.nlayers = nlayers
        self.lmap_size = lmap_size
        self.im_size = lmap_size * self.ds
        self.multigpu_full = multigpu_full
        self.bg_bias = bg_bias
        self.normalize = normalize
        self.debug_mode = debug_mode

        # dictionary
        self.dictionary = dictionary

        ########## modules ##########
        # proposal networks
        self.reader = Reader(indim=3, hiddim=hiddim, outdim=hiddim, ds_times=self.downsample, normalize=normalize,
                             nlayers=nlayers)
        self.h_mean = nn.Conv2d(hiddim, latentdim, 3, 1, 1)
        self.h_var = nn.Conv2d(hiddim, latentdim, 3, 1, 1)

        # pixel writer
        self.writer = Writer(indim=latentdim, hiddim=hiddim, outdim=3, ds_times=self.downsample, normalize=normalize,
                             nlayers=nlayers)

        # visual words
        self.vis_dist = ConceptMapper(word_size, len(dictionary))
        self.pos_dist = ConceptMapper(pos_size, len(dictionary))

        self.renderer = DistributionRender(hiddim=latentdim)

        # neural modules
        self.combine = Combine(hiddim_v=latentdim, hiddim_p=pos_size[0], op=op[0])
        self.describe = Describe(hiddim_v=latentdim, hiddim_p=pos_size[0], op=op[1])
        self.transform = Transform(matrix='default')

        # small vaes for bounding boxes and offsets learning
        # input: H, W
        self.box_vae = VAE(indim=2, latentdim=pos_size[0])
        # input: [x0, y0, x1, y1] + condition: [H0, W0, H1, W1, im_H, im_W, (im_H-H0), (im_W-W0), (im_H-H1), (im_W-W1)]
        self.offset_vae = VAE(indim=4, latentdim=pos_size[0])

        ## loss functions&sampler
        self.sampler = reparameterize()
        if lambdakl > 0:
            from lib.LambdaBiKLD import BiKLD
            self.bikld = BiKLD(lambda_t=lambdakl, k=None)
        else:
            from lib.BiKLD import BiKLD
            self.bikld = BiKLD()

        if loss == 'l1':
            self.pixelrecon_criterion = nn.L1Loss()
        elif loss == 'l2':
            self.pixelrecon_criterion = nn.MSELoss()
        self.pixelrecon_criterion.size_average = False

        self.pos_criterion = nn.MSELoss()
        self.pos_criterion.size_average = False

        ## biases
        self.bias_mean = nn.Linear(1, self.latentdim * self.lmap_size * self.lmap_size, bias=False)
        self.bias_var = nn.Linear(1, self.latentdim * self.lmap_size * self.lmap_size, bias=False)
        self.latent_canvas_size = torch.Size([1, self.latentdim, self.lmap_size, self.lmap_size])

    def get_mask_from_tree(self, tree, size):
        mask = Variable(torch.zeros(size), requires_grad=False).cuda()

        return self._get_mask_from_tree(tree, mask)

    def _get_mask_from_tree(self, tree, mask):
        for i in range(0, tree.num_children):
            mask = self._get_mask_from_tree(tree.children[i], mask)

        if tree.function == 'describe':
            bbx = tree.bbox
            mask[:, :, bbx[0]:bbx[0] + bbx[2], bbx[1]:bbx[1] + bbx[3]] = 1.0

        return mask

    def forward(self, x, treex, treeindex=None, alpha=1.0, ifmask=False, maskweight=1.0):
        ################################
        ##    input: images, trees    ##
        ################################

        # if multigpu_full, pick the trees by treeindex
        if self.multigpu_full:
            treex_pick = [treex[ele[0]] for ele in treeindex.data.cpu().numpy().astype(int)]
            treex = treex_pick

        if ifmask == True:
            mask = []
            for i in range(0, len(treex)):
                mask += [self.get_mask_from_tree(treex[i], x[0:1, :, :, :].size())]
            mask = torch.cat(mask, dim=0)

        # encoding the images
        h = self.reader(x)
        # proposal distribution
        latent_mean = self.h_mean(h)
        latent_var = self.h_var(h)

        # losses
        kld_loss, rec_loss, pos_loss = 0, 0, 0

        # forward GNMN
        prior_mean_all = []
        prior_var_all = []
        trees = []
        for i in range(0, len(treex)):  # iterate through every tree of the batch
            trees.append(self.compose_tree(treex[i], self.latent_canvas_size))
            prior_mean_all += [trees[i].vis_dist[0]]
            prior_var_all += [trees[i].vis_dist[1]]
            pos_loss += trees[i].pos_loss
            if np.isnan(trees[i].pos_loss.data.cpu().numpy()):
                print('found nan pos loss')
                import IPython;
                IPython.embed()

        prior_mean = torch.cat(prior_mean_all, dim=0)
        prior_var = torch.cat(prior_var_all, dim=0)

        prior_mean, prior_var = self.renderer([prior_mean, prior_var])

        # sample z map
        z_map = self.sampler(latent_mean, latent_var)

        # kld loss
        kld_loss = alpha * self.bikld([latent_mean, latent_var], [prior_mean, prior_var]) + \
                   (1 - alpha) * self.bikld([latent_mean.detach(), latent_var.detach()], [prior_mean, prior_var])

        rec = self.writer(z_map)

        if ifmask is True:
            mask = (mask + maskweight) / (maskweight + 1.0)
            rec_loss = self.pixelrecon_criterion(mask * rec, mask * x)
        else:
            rec_loss = self.pixelrecon_criterion(rec, x)
        rec_loss = rec_loss.sum()

        return rec_loss, kld_loss, pos_loss, rec

    def get_code(self, dictionary, word):
        code = Variable(torch.zeros(1, len(dictionary))).cuda()
        code[0, dictionary.index(word)] = 1

        return code

    def compose_tree(self, treex, latent_canvas_size):
        for i in range(0, treex.num_children):
            treex.children[i] = self.compose_tree(treex.children[i], latent_canvas_size)

        # one hot embedding of a word
        ohe = self.get_code(self.dictionary, treex.word)

        if treex.function == 'combine':
            vis_dist = self.vis_dist(ohe)
            pos_dist = self.pos_dist(ohe)
            if treex.num_children > 0:
                # visual content
                vis_dist_child = treex.children[0].vis_dist
                vis_dist = self.combine(vis_dist, vis_dist_child, 'vis')
                # visual position
                pos_dist_child = treex.children[0].pos_dist
                pos_dist = self.combine(pos_dist, pos_dist_child, 'pos')

            treex.vis_dist = vis_dist
            treex.pos_dist = pos_dist

        elif treex.function == 'describe':
            # blend visual words
            vis_dist = self.vis_dist(ohe)
            pos_dist = self.pos_dist(ohe)
            if treex.num_children > 0:
                # visual content
                vis_dist_child = treex.children[0].vis_dist
                vis_dist = self.describe(vis_dist_child, vis_dist, 'vis')
                # visual position
                pos_dist_child = treex.children[0].pos_dist
                pos_dist = self.describe(pos_dist_child, pos_dist, 'pos')

            treex.pos_dist = pos_dist

            # regress bbox
            treex.pos = np.maximum(treex.bbox[2:] // self.ds, [1, 1])
            target_box = Variable(torch.from_numpy(np.array(treex.bbox[2:])[np.newaxis, ...].astype(np.float32))).cuda()
            regress_box, kl_box = self.box_vae(target_box, prior=treex.pos_dist)
            treex.pos_loss = self.pos_criterion(regress_box, target_box) + kl_box

            if treex.parent == None:
                ones = self.get_ones(torch.Size([1, 1]))
                if not self.bg_bias:
                    bg_vis_dist = [Variable(torch.zeros(latent_canvas_size)).cuda(), \
                                   Variable(torch.zeros(latent_canvas_size)).cuda()]
                else:
                    bg_vis_dist = [self.bias_mean(ones).view(*latent_canvas_size), \
                                   self.bias_var(ones).view(*latent_canvas_size)]
                b = np.maximum(treex.bbox // self.ds, [0, 0, 1, 1])

                bg_vis_dist = [self.assign_util(bg_vis_dist[0], b, self.transform(vis_dist[0], treex.pos),
                                                'assign'), \
                               self.assign_util(bg_vis_dist[1], b,
                                                self.transform(vis_dist[1], treex.pos, variance=True),
                                                'assign')]
                vis_dist = bg_vis_dist
            else:
                try:
                    # resize vis_dist
                    vis_dist = [self.transform(vis_dist[0], treex.pos), \
                                self.transform(vis_dist[1], treex.pos, variance=True)]
                except:
                    import IPython;
                    IPython.embed()

            treex.vis_dist = vis_dist

        elif treex.function == 'layout':
            # get pos word as position prior
            treex.pos_dist = self.pos_dist(ohe)
            assert (treex.num_children > 0)

            # get offsets: use gt for training
            l_pos = treex.children[0].pos
            l_offset = np.maximum(treex.children[0].bbox[:2] // self.ds, [1, 1])

            r_pos = treex.children[1].pos
            r_offset = np.maximum(treex.children[1].bbox[:2] // self.ds, [1, 1])

            # regress offsets
            target_offset = np.append(l_offset * self.ds, r_offset * self.ds).astype(np.float32)
            target_offset = Variable(torch.from_numpy(target_offset[np.newaxis, ...])).cuda()
            regress_offset, kl_offset = self.offset_vae(target_offset, prior=treex.pos_dist)
            treex.pos_loss = self.pos_criterion(regress_offset, target_offset) + kl_offset + treex.children[
                0].pos_loss + \
                             treex.children[1].pos_loss

            ######################### constructing latent map ###############################
            # bias filled mean&var
            ones = self.get_ones(torch.Size([1, 1]))
            if not self.bg_bias:
                vis_dist = [Variable(torch.zeros(latent_canvas_size)).cuda(), \
                            Variable(torch.zeros(latent_canvas_size)).cuda()]
            else:
                vis_dist = [self.bias_mean(ones).view(*latent_canvas_size), \
                            self.bias_var(ones).view(*latent_canvas_size)]

            # arrange the layout of two children
            vis_dist[0] = self.assign_util(vis_dist[0], list(l_offset) + list(l_pos), treex.children[0].vis_dist[0],
                                           'assign')
            vis_dist[1] = self.assign_util(vis_dist[1], list(l_offset) + list(l_pos), treex.children[0].vis_dist[1],
                                           'assign')

            vis_dist[0] = self.assign_util(vis_dist[0], list(r_offset) + list(r_pos), treex.children[1].vis_dist[0],
                                           'assign')
            vis_dist[1] = self.assign_util(vis_dist[1], list(r_offset) + list(r_pos), treex.children[1].vis_dist[1],
                                           'assign')

            # continue layout
            if treex.parent != None:
                p = [min(l_offset[0], r_offset[0]), min(l_offset[1], r_offset[1]), \
                     max(l_offset[0] + l_pos[0], r_offset[0] + r_pos[0]),
                     max(l_offset[1] + l_pos[1], r_offset[1] + r_pos[1])]
                treex.pos = [p[2] - p[0], p[3] - p[1]]
                treex.vis_dist = [vis_dist[0][:, :, p[0]:p[2], p[1]:p[3]], \
                                  vis_dist[1][:, :, p[0]:p[2], p[1]:p[3]]]
            else:
                treex.vis_dist = vis_dist

        return treex

    def assign_util(self, a, bx, b, mode):
        if mode == 'assign':
            a[:, :, bx[0]:bx[0] + bx[2], bx[1]:bx[1] + bx[3]] = b
        elif mode == 'add':
            a[:, :, bx[0]:bx[0] + bx[2], bx[1]:bx[1] + bx[3]] = \
                a[:, :, bx[0]:bx[0] + bx[2], bx[1]:bx[1] + bx[3]] + b
        elif mode == 'slice':
            a = a[:, :, bx[0]:bx[0] + bx[2], bx[1]:bx[1] + bx[3]].clone()
        else:
            raise ValueError('Please specify the correct mode.')
        return a

    def overlap_box(self, box_left, box_right):
        x1, y1, h1, w1 = box_left[0], box_left[1], box_left[2], box_left[3]
        x2, y2, h2, w2 = box_right[0], box_right[1], box_right[2], box_right[3]

        ox1 = max(x1, x2)
        oy1 = max(y1, y2)
        ox2 = min(x1 + h1, x2 + h2)
        oy2 = min(y1 + w1, y2 + w2)

        if ox2 > ox1 and oy2 > oy1:
            return [ox1, oy1, ox2 - ox1, oy2 - oy1]
        else:
            return []

    def generate(self, x, treex, treeindex=None):
        ################################
        ##    input: images, trees    ##
        ################################
        if self.multigpu_full:
            treex_pick = [treex[ele[0]] for ele in treeindex.data.cpu().numpy().astype(int)]
            treex = treex_pick

        # tranverse trees to compose visual words
        prior_mean = []
        prior_var = []

        for i in range(0, len(treex)):
            treex[i] = self.generate_compose_tree(treex[i], self.latent_canvas_size)
            prior_mean += [treex[i].vis_dist[0]]
            prior_var += [treex[i].vis_dist[1]]
        prior_mean = torch.cat(prior_mean, dim=0)
        prior_var = torch.cat(prior_var, dim=0)

        # sample z map
        prior_mean, prior_var = self.renderer([prior_mean, prior_var])

        z_map = self.sampler(prior_mean, prior_var)

        rec = self.writer(z_map)

        return rec

    def check_valid(self, offsets, l_pos, r_pos, im_size):
        flag = True
        if offsets[0] + l_pos[0] > im_size:
            flag = False
            return flag
        if offsets[1] + l_pos[1] > im_size:
            flag = False
            return flag
        if offsets[2] + r_pos[0] > im_size:
            flag = False
            return flag
        if offsets[3] + r_pos[1] > im_size:
            flag = False
            return flag

        return flag

    def generate_compose_tree(self, treex, latent_canvas_size):
        for i in range(0, treex.num_children):
            treex.children[i] = self.generate_compose_tree(treex.children[i], latent_canvas_size)

        # one hot embedding of a word
        ohe = self.get_code(self.dictionary, treex.word)
        if treex.function == 'combine':
            vis_dist = self.vis_dist(ohe)
            pos_dist = self.pos_dist(ohe)
            if treex.num_children > 0:
                # visual content
                vis_dist_child = treex.children[0].vis_dist
                vis_dist = self.combine(vis_dist, vis_dist_child, 'vis')
                # visual position
                pos_dist_child = treex.children[0].pos_dist
                pos_dist = self.combine(pos_dist, pos_dist_child, 'pos')

            treex.vis_dist = vis_dist
            treex.pos_dist = pos_dist

        elif treex.function == 'describe':
            # blend visual words
            vis_dist = self.vis_dist(ohe)
            pos_dist = self.pos_dist(ohe)
            if treex.num_children > 0:
                # visual content
                vis_dist_child = treex.children[0].vis_dist
                vis_dist = self.describe(vis_dist_child, vis_dist, 'vis')
                # visual position
                pos_dist_child = treex.children[0].pos_dist
                pos_dist = self.describe(pos_dist_child, pos_dist, 'pos')

            treex.pos_dist = pos_dist

            # regress bbox
            treex.pos = np.clip(self.box_vae.generate(prior=treex.pos_dist).data.cpu().numpy().astype(int),
                                int(self.ds),
                                self.im_size).flatten() // self.ds

            if treex.parent == None:
                ones = self.get_ones(torch.Size([1, 1]))
                if not self.bg_bias:
                    bg_vis_dist = [Variable(torch.zeros(latent_canvas_size)).cuda(), \
                                   Variable(torch.zeros(latent_canvas_size)).cuda()]
                else:
                    bg_vis_dist = [self.bias_mean(ones).view(*latent_canvas_size), \
                                   self.bias_var(ones).view(*latent_canvas_size)]
                b = [int(latent_canvas_size[2]) // 2 - treex.pos[0] // 2,
                     int(latent_canvas_size[3]) // 2 - treex.pos[1] // 2, treex.pos[0], treex.pos[1]]

                bg_vis_dist = [self.assign_util(bg_vis_dist[0], b, self.transform(vis_dist[0], treex.pos),
                                                'assign'), \
                               self.assign_util(bg_vis_dist[1], b,
                                                self.transform(vis_dist[1], treex.pos, variance=True),
                                                'assign')]

                vis_dist = bg_vis_dist
                treex.offsets = b
            else:
                # resize vis_dist
                vis_dist = [self.transform(vis_dist[0], treex.pos), \
                            self.transform(vis_dist[1], treex.pos, variance=True)]

            treex.vis_dist = vis_dist

        elif treex.function == 'layout':
            # get pos word as position prior
            treex.pos_dist = self.pos_dist(ohe)
            assert (treex.num_children > 0)

            # get offsets: use gt for training
            l_pos = treex.children[0].pos
            r_pos = treex.children[1].pos

            offsets = np.clip(self.offset_vae.generate(prior=treex.pos_dist).data.cpu().numpy().astype(int), 0,
                              self.im_size).flatten() // self.ds
            countdown = 0
            while self.check_valid(offsets, l_pos, r_pos, self.im_size // self.ds) == False:
                offsets = np.clip(self.offset_vae.generate(prior=treex.pos_dist).data.cpu().numpy().astype(int), 0,
                                  self.im_size).flatten() // self.ds
                if countdown >= 100:
                    print('Tried proposing more than 100 times.')
                    if self.debug_mode:
                        import IPython;
                        IPython.embed()
                    print('Warning! Manually adapt offsets')
                    lat_size = self.im_size // self.ds
                    if offsets[0] + l_pos[0] > lat_size:
                        offsets[0] = lat_size - l_pos[0]
                    if offsets[1] + l_pos[1] > lat_size:
                        offsets[1] = lat_size - l_pos[1]
                    if offsets[2] + r_pos[0] > lat_size:
                        offsets[2] = lat_size - r_pos[0]
                    if offsets[3] + r_pos[1] > lat_size:
                        offsets[3] = lat_size - r_pos[1]

                countdown += 1
            treex.offsets = offsets
            l_offset = offsets[:2]
            r_offset = offsets[2:]

            ######################### constructing latent map ###############################
            # bias filled mean&var
            ones = self.get_ones(torch.Size([1, 1]))
            if not self.bg_bias:
                bg_vis_dist = [Variable(torch.zeros(latent_canvas_size)).cuda(), \
                               Variable(torch.zeros(latent_canvas_size)).cuda()]
            else:
                bg_vis_dist = [self.bias_mean(ones).view(*latent_canvas_size), \
                               self.bias_var(ones).view(*latent_canvas_size)]

            vis_dist = bg_vis_dist
            try:
                # arrange the layout of two children
                vis_dist[0] = self.assign_util(vis_dist[0], list(l_offset) + list(l_pos), treex.children[0].vis_dist[0],
                                               'assign')
                vis_dist[1] = self.assign_util(vis_dist[1], list(l_offset) + list(l_pos), treex.children[0].vis_dist[1],
                                               'assign')

                vis_dist[0] = self.assign_util(vis_dist[0], list(r_offset) + list(r_pos), treex.children[1].vis_dist[0],
                                               'assign')
                vis_dist[1] = self.assign_util(vis_dist[1], list(r_offset) + list(r_pos), treex.children[1].vis_dist[1],
                                               'assign')
            except:
                print('latent distribution doesnt fit size.')
                import IPython;
                IPython.embed()

            if treex.parent != None:
                p = [min(l_offset[0], r_offset[0]), min(l_offset[1], r_offset[1]), \
                     max(l_offset[0] + l_pos[0], r_offset[0] + r_pos[0]),
                     max(l_offset[1] + l_pos[1], r_offset[1] + r_pos[1])]
                treex.pos = [p[2] - p[0], p[3] - p[1]]
                treex.vis_dist = [vis_dist[0][:, :, p[0]:p[2], p[1]:p[3]], \
                                  vis_dist[1][:, :, p[0]:p[2], p[1]:p[3]]]
            else:
                treex.vis_dist = vis_dist

        return treex

    def get_ones(self, size):
        return Variable(torch.ones(size), requires_grad=False).cuda()

    def clean_tree(self, treex):
        for i in range(0, len(treex)):
            self._clean_tree(treex[i])

    def _clean_tree(self, treex):
        for i in range(0, treex.num_children):
            self._clean_tree(treex.children[i])

        if treex.function == 'combine':
            treex.vis_dist = None
            treex.pos_dist = None
        elif treex.function == 'describe':
            treex.vis_dist = None
            treex.pos_dist = None
            treex.pos = None
        elif treex.function == 'layout':
            treex.vis_dist = None
            treex.pos_dist = None
            treex.pos = None
