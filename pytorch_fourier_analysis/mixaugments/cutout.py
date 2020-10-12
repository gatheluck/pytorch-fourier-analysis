import math
import random

import numpy as np
import torch

from .base import MixAugmentationBase


class Cutout(MixAugmentationBase):
    def __init__(self, prob: float, cutout_size: int):
        """
        Cutout (https://arxiv.org/abs/1708.04552) imprementation.

        Args
            prob: Probability of using Mixup. This param is set as 0.5 for CIFAR and 1.0 for ImageNet in the CutMix paper.
            cutout_size: Size of Cutout region. In the original paper, 16 for CIFAR-10, 8 for CIFAR-100.
        """
        self.prob = prob
        self.cutout_size = cutout_size

    def __call__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        x: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Args
            model: Model used for prediction.
            criterion: Function which map logits to loss.
            x: Input tensor. Shape should be [b, c, h, w].
            t: Target classes. Shape should be [b].
            criterion: Function which map from logits to loss.
        """
        r = np.random.rand(1)
        if r < self.prob:
            zeros = torch.zeros_like(x)
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), self.cutout_size)
            x[:, :, bbx1:bbx2, bby1:bby2] = zeros[:, :, bbx1:bbx2, bby1:bby2]

        # compute output
        output = model(x)
        loss = criterion(output, t)

        retdict = dict(x=x.detach(), output=output.detach(), loss=loss.detach())
        return loss, retdict

    def _rand_bbox(self, size: torch.Size, cutout_size: int):
        w, h = size[2], size[3]

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - (cutout_size // 2), 0, w)
        bby1 = np.clip(cy - (cutout_size // 2), 0, h)
        bbx2 = np.clip(cx + (cutout_size // 2), 0, w)
        bby2 = np.clip(cy + (cutout_size // 2), 0, h)

        return bbx1, bby1, bbx2, bby2
