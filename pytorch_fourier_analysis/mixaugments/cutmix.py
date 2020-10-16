import numpy as np
import torch

from .base import MixAugmentationBase


class CutMix(MixAugmentationBase):
    def __init__(self, alpha: float, prob: float):
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        x: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        CutMix (https://arxiv.org/abs/1905.04899) imprementation.
        Some parts of code is borrowed from https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py.

        Args
            model: Model used for prediction.
            x: Input tensor. Shape should be [b, c, h, w].
            t: Target classes. Shape should be [b].
            criterion: Function which map from logits to loss.
            alpha: Lamda is sampled from beta distribution Beta(alpha, alpha). This param is set as 1.0 in the original paper.
            prob: Probability of using CutMix. This param is set as 0.5 for CIFAR and 1.0 for ImageNet in the original paper.
        """
        r = np.random.rand(1)
        if self.alpha > 0 and r < self.prob:
            # generate mixed sample
            lam = np.random.beta(self.alpha, self.alpha)
            rand_index = torch.randperm(x.size(0))
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

            # compute output
            output = model(x)
            loss = self._calc_mixed_loss(output, t, rand_index, lam, criterion)
        else:
            # compute output
            output = model(x)
            loss = criterion(output, t)

        retdict = dict(x=x.detach(), output=output.detach(), loss=loss.detach())
        return loss, retdict

    def _rand_bbox(self, size: torch.Size, lam: float):
        w = size[2]
        h = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2
