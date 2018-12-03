import scipy.misc
import numpy as np


def color_grid_vis(X, nh, nw, save_path):
    h, w = X[0].shape[:2]
    img = np.zeros((h * nh, w * nw, 3))
    for n, x in enumerate(X):
        j = int(n / nw)
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w, :] = x
    scipy.misc.imsave(save_path, img)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.pixel_count = 0
        self.batch_count = 0

    def update(self, val, n=1, batch=1):
        self.val = val
        self.sum += val * n
        self.pixel_count += n
        self.batch_count += batch
        self.pixel_avg = self.sum / self.pixel_count
        self.batch_avg = self.sum / self.batch_count
