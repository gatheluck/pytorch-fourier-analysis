import random

import numpy as np
import torch

from .base import NoiseAugmentationBase


class Gaussian(NoiseAugmentationBase):
    def __init__(self, prob: float, max_scale: float, randomize_scale: bool):
        """
        Args
            prob: Probability of using Patch Gaussian
            max_scale: Max scale of Gaussian noise
            randomize_scale: Randomizing scale or not
        """
        self.prob = prob
        self.max_scale = max_scale
        self.randomize_scale = randomize_scale

    def __call__(self, x: torch.Tensor):
        r = np.random.rand(1)
        if r < self.prob:
            c, h, w = x.shape[-3:]
            scale = (
                random.uniform(0, 1) * self.max_scale
                if self.randomize_scale
                else self.max_scale
            )
            gaussian = torch.normal(mean=0.0, std=scale, size=(c, h, w))
            return torch.clamp(x + gaussian, 0.0, 1.0)
        else:
            return x


class PatchGaussian(NoiseAugmentationBase):
    def __init__(
        self,
        prob: float,
        patch_size: int,
        randomize_patch_size: bool,
        max_scale: float,
        randomize_scale: bool,
    ):
        """
        Args
            prob: Probability of using Patch Gaussian
            patch_size: Size of patch. In the original paper, 25 for CIFAR-10 and 224 for ImageNet
            randomize_patch_size: Randomizing patch size or not
            max_scale: Max scale of Gaussian noise
            randomize_scale: Randomizing scale or not
        """
        self.prob = prob
        self.patch_size = patch_size
        self.randomize_patch_size = randomize_patch_size
        self.max_scale = max_scale
        self.randomize_scale = randomize_scale

    def __call__(self, x: torch.Tensor):
        r = np.random.rand(1)
        if r < self.prob:
            c, h, w = x.shape[-3:]
            # generate noise
            scale = (
                random.uniform(0, 1) * self.max_scale
                if self.randomize_scale
                else self.max_scale
            )
            gaussian = torch.normal(mean=0.0, std=scale, size=(c, h, w))

            # generate mask
            patch_size = (
                random.randrange(1, self.patch_size + 1)
                if self.randomize_patch_size
                else self.patch_size
            )
            mask = self.sample_mask(x.size(), patch_size)

            return self.add_masked_noise(x, gaussian, mask)
        else:
            return x
