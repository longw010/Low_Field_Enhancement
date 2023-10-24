

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
import time
import logging
import yaml


def set_seed(seed: int):
    """
    platform agnostic seed
    :return:
    """
    # note that this still won't be entirely deterministic
    # a better solution can be found at https://github.com/NVIDIA/framework-determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_seed_torch(seed: int):
    """ 100% deterministically """
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class SSIMLoss(torch.nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.register_buffer("kernel", self._cal_gaussian_kernel(11, 1.5))
        self.L = 2.0
        self.k1 = 0.01
        self.k2 = 0.03

    @staticmethod
    def _cal_gaussian_kernel(size, sigma):
        g = torch.Tensor([math.exp(-(x - size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(size)])
        g = g / g.sum()
        window = g.reshape([-1, 1]).matmul(g.reshape([1, -1]))
        #kernel = torch.reshape(window, [1, 1, size, size]).repeat(3, 1, 1, 1)
        kernel = torch.reshape(window, [1, 1, size, size])
        return kernel

    def forward(self, img0, img1):
        """
        :param img0: range in (-1, 1)
        :param img1: range in (-1, 1)
        :return: SSIM loss i.e. 1 - ssim
        """
        mu0 = torch.nn.functional.conv2d(img0, self.kernel, padding=0, groups=1)
        mu1 = torch.nn.functional.conv2d(img1, self.kernel, padding=0, groups=1)
        mu0_sq = torch.pow(mu0, 2)
        mu1_sq = torch.pow(mu1, 2)
        var0 = torch.nn.functional.conv2d(img0 * img0, self.kernel, padding=0, groups=1) - mu0_sq
        var1 = torch.nn.functional.conv2d(img1 * img1, self.kernel, padding=0, groups=1) - mu1_sq
        covar = torch.nn.functional.conv2d(img0 * img1, self.kernel, padding=0, groups=1) - mu0 * mu1
        c1 = (self.k1 * self.L) ** 2
        c2 = (self.k2 * self.L) ** 2
        ssim_numerator = (2 * mu0 * mu1 + c1) * (2 * covar + c2)
        ssim_denominator = (mu0_sq + mu1_sq + c1) * (var0 + var1 + c2)
        ssim = ssim_numerator / ssim_denominator
        ssim_loss = 1.0 - ssim
        return ssim_loss
