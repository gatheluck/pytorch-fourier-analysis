import numpy as np
import torch
from .base import MixAugmentationBase


class Mixup(MixAugmentationBase):
    def __init__(self, alpha: float, prob: float):
        """
        Mixup (https://arxiv.org/abs/1710.09412) imprementation.
        Some parts of code is borrowed from https://github.com/facebookresearch/mixup-cifar10.

        Args
            alpha: Lamda is sampled from beta distribution Beta(alpha, alpha). This param is set as 1.0 in the original paper.
            prob: Probability of using Mixup. This param is set as 0.5 for CIFAR and 1.0 for ImageNet in the CutMix paper.
        """
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
        Args
            model: Model used for prediction.
            criterion: Function which map logits to loss.
            x: Input tensor. Shape should be [b, c, h, w].
            t: Target classes. Shape should be [b].
            criterion: Function which map from logits to loss.
        """
        r = np.random.rand(1)
        if self.alpha > 0 and r < self.prob:
            # generate mixed sample
            lam = np.random.beta(self.alpha, self.alpha)
            rand_index = torch.randperm(x.size(0))
            x = (lam * x) + ((1.0 - lam) * x[rand_index, :])

            # compute output
            output = model(x)
            loss = self._calc_mixed_loss(output, t, rand_index, lam, criterion)
        else:
            # compute output
            output = model(x)
            loss = criterion(output, t)

        retdict = dict(x=x.detach(), output=output.detach(), loss=loss.detach())
        return loss, retdict
