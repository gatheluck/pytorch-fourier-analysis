import os
import sys
import random
from typing import Optional, Tuple, Union
from typing_extensions import Literal

import numpy as np
import torch

from .base import NoiseAugmentationBase

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import fourier


FilterMode = Literal[None, "low_pass", "high_pass"]


def get_gaussian_noise(
    mean: Union[float, torch.Tensor],
    std: Union[float, torch.Tensor],
    size: Tuple[int],
    mode: FilterMode,
    bandwidth: Optional[int],
    adjust_eps: Optional[bool],
) -> torch.Tensor:
    """
    return gaussian noise with bandpass filter.
    """
    gaussian = torch.normal(mean=mean, std=std, size=size)  # (c,h,w)
    if not (mode and bandwidth):
        return gaussian

    # apply bandpass filter
    filtered_gaussian, _ = fourier.bandpass_filter(
        gaussian.unsqueeze(0), bandwidth, mode, None
    )
    filtered_gaussian = filtered_gaussian.squeeze(0)
    if not adjust_eps:
        return filtered_gaussian

    else:
        # adjust eps
        eps = gaussian.view(gaussian.size(0), -1).norm(dim=-1)  # (c)
        eps_filtered = filtered_gaussian.view(gaussian.size(0), -1).norm(dim=-1)  # (c)
        filtered_gaussian /= eps_filtered[:, None, None]
        return filtered_gaussian * eps[:, None, None]


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


class BandpassGaussian(NoiseAugmentationBase):
    def __init__(
        self,
        prob: float,
        max_scale: float,
        randomize_scale: bool,
        filter_mode: FilterMode,
    ):
        self.prob = prob
        self.max_scale = max_scale
        self.randomize_scale = randomize_scale
        self.filter_mode = filter_mode

    def __call__(self, x: torch.Tensor):
        r = np.random.rand(1)
        if r < self.prob:
            c, h, w = x.shape[-3:]
            scale = (
                random.uniform(0, 1) * self.max_scale
                if self.randomize_scale
                else self.max_scale
            )

            bandwidth = random.randrange(1, min(h, w), 2)

            filtered_gaussian = get_gaussian_noise(
                mean=0.0,
                std=scale,
                size=(c, h, w),
                mode=self.filter_mode,
                bandwidth=bandwidth,
                adjust_eps=True,
            )
            return torch.clamp(x + filtered_gaussian, 0.0, 1.0)
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
