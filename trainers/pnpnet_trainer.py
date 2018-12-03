import scipy.misc
import numpy as np
import os
import os.path as osp
import datetime
import pytz
import pdb

import torch
from torch.autograd import Variable

from lib.utils import color_grid_vis, AverageMeter


class PNPNetTrainer:
    def __init__(self, model, train_loader, val_loader, gen_loader, optimizer, configs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gen_loader = gen_loader
        self.optimizer = optimizer
        self.configs = configs

    def train_epoch(self, epoch_num, timestamp_start):
        self.model.train()
        train_rec_loss = AverageMeter()
        train_kld_loss = AverageMeter()
        train_pos_loss = AverageMeter()
        batch_idx = 0
        epoch_end = False
        # annealing for kl penalty
        kl_coeff = float(epoch_num) / float(self.configs.warmup_iter + 1)
        if kl_coeff >= self.configs.alpha_ub:
            kl_coeff = self.configs.alpha_ub
        print('kl penalty coefficient: ', kl_coeff, 'alpha upperbound:', self.configs.alpha_ub)
        while epoch_end is False:
            data, trees, _, epoch_end = self.train_loader.next_batch()
            data = Variable(data).cuda()

            self.optimizer.zero_grad()
            ifmask = False
            if self.configs.maskweight > 0:
                ifmask = True
            rec_loss, kld_loss, pos_loss, modelout = self.model(data, trees, alpha=kl_coeff, ifmask=ifmask,
                                                                maskweight=self.configs.maskweight)
            recon = modelout
            rec_loss, kld_loss, pos_loss = rec_loss.sum() / self._total(data), kld_loss.sum() / self._total(data), pos_loss.sum() / self._total(data)
            loss = rec_loss + self.configs.kl_beta * kld_loss + self.configs.pos_beta * pos_loss
            loss.backward()
            self.optimizer.step()
            train_rec_loss.update(rec_loss.item(), self._total(data), data.size(0))
            train_kld_loss.update(kld_loss.item(), self._total(data), data.size(0))
            train_pos_loss.update(pos_loss.item(), self._total(data), data.size(0))

            if batch_idx % 30 == 0:
                scipy.misc.imsave(osp.join(self.configs.exp_dir, 'samples', 'generativenmn_data.png'),
                                  (data.cpu().data.numpy().transpose(0, 2, 3, 1)[0] + 1) / 2.0)
                scipy.misc.imsave(osp.join(self.configs.exp_dir, 'samples', 'generativenmn_reconstruction.png'), \
                                  (recon.cpu().data.numpy().transpose(0, 2, 3, 1)[0] + 1) / 2.0)
                scipy.misc.imsave(osp.join(self.configs.exp_dir, 'samples', 'generativenmn_reconstruction_clip.png'), \
                                  np.clip(recon.cpu().data.numpy().transpose(0, 2, 3, 1)[0], -1, 1))
                print('Epoch:{0}\tIter:{1}/{2}\tRecon {3:.6f}\t KL {4:.6f}'.format(epoch_num, batch_idx,
                                                                                   len(self.train_loader) // self.configs.batch_size,
                                                                                   train_rec_loss.batch_avg, train_kld_loss.batch_avg))

            self.model.clean_tree(trees)
            batch_idx += 1

        elapsed_time = \
            datetime.datetime.now(pytz.timezone('America/New_York')) - \
            timestamp_start

        print('====> Epoch: {}  Average rec loss: {:.6f} Average kld loss: {:.6f} Average pos loss: {:.6f}'.format(
            epoch_num, train_rec_loss.batch_avg, train_kld_loss.batch_avg, train_pos_loss.batch_avg))
        print('Elapsed time:', elapsed_time)

    @staticmethod
    def _total(tensor):
        return tensor.size(0) * tensor.size(1) * tensor.size(2) * tensor.size(3)

    def validate(self, epoch_num, timestamp_start, minloss):
        self.model.eval()
        test_rec_loss = AverageMeter()
        test_kld_loss = AverageMeter()
        test_pos_loss = AverageMeter()
        epoch_end = False
        count = 0.
        while epoch_end is False:
            data, trees, _, epoch_end = self.val_loader.next_batch()
            data = Variable(data, volatile=True).cuda()
            rec_loss, kld_loss, pos_loss, modelout = self.model(data, trees)

            rec_loss, kld_loss, pos_loss = rec_loss.sum(), kld_loss.sum(), pos_loss.sum()
            loss = rec_loss + kld_loss + pos_loss

            test_rec_loss.update(rec_loss.item(), self._total(data), data.size(0))
            test_kld_loss.update(kld_loss.item(), self._total(data), data.size(0))
            test_pos_loss.update(pos_loss.item(), self._total(data), data.size(0))

            self.model.clean_tree(trees)

        elapsed_time = \
            datetime.datetime.now(pytz.timezone('America/New_York')) - \
            timestamp_start

        torch.save(self.model.state_dict(),
                   osp.join(self.configs.exp_dir, 'checkpoints', 'model_epoch_{0}.pth'.format(epoch_num)))

        print('====> Epoch: {}  Test rec loss: {:.6f} Test kld loss: {:.6f} Test pos loss: {:.6f}'.format(
            epoch_num, test_rec_loss.batch_avg, test_kld_loss.batch_avg, test_pos_loss.batch_avg))
        print('Elapsed time:', elapsed_time)

        return minloss

    def sample(self, epoch_num, sample_num, timestamp_start):
        self.model.eval()

        data, trees, _, _ = self.gen_loader.next_batch()
        data = Variable(data, volatile=True).cuda()
        epoch_result_dir = osp.join(self.configs.exp_dir, 'samples', 'epoch-{}'.format(epoch_num))

        try:
            os.makedirs(epoch_result_dir)
        except:
            pass

        samples_image_dict = dict()
        data_image_dict = dict()
        batch_size = None
        for j in range(sample_num):
            sample = self.model.generate(data, trees)
            if not batch_size:
                batch_size = sample.size(0)
            for i in range(0, sample.size(0)):
                samples_image_dict.setdefault(i, list()).append(sample.cpu().data.numpy().transpose(0, 2, 3, 1)[i])
                if j == sample_num - 1:
                    data_image_dict[i] = data.cpu().data.numpy().transpose(0, 2, 3, 1)[i]
            self.model.clean_tree(trees)
            print(j)

        for i in range(batch_size):
            samples = np.clip(np.stack(samples_image_dict[i], axis=0), -1, 1)
            data = data_image_dict[i]
            color_grid_vis(samples, nh=2, nw=sample_num // 2,
                           save_path=osp.join(epoch_result_dir, 'generativenmn_{}_sample.png'.format(i)))
            scipy.misc.imsave(osp.join(epoch_result_dir, 'generativenmn_{}_real.png'.format(i)), data)
            torch.save(trees[i], osp.join(epoch_result_dir, 'generativenmn_tree_' + str(i) + '.pth'))
            print('====> Epoch: {}  Generating image number: {:d}'.format(epoch_num, i))

        elapsed_time = \
            datetime.datetime.now(pytz.timezone('America/New_York')) - \
            timestamp_start
        print('Elapsed time:', elapsed_time)
