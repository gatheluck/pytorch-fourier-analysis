import random

import numpy as np
import torch

from .base import NoiseAugmentationBase
import pytorch_fourier_analysis.fourier.basis


class Fourier(NoiseAugmentationBase):
    def __init__(self, prob: float, max_index: int, max_eps: float):
        """
        Args
            prob: Probability of using Patch Gaussian
            max_index: Maximun index of Fourier basis
            max_eps: Maximun noise size in l2 norm
        """
        self.prob = prob
        self.max_index = max_index
        self.max_eps = max_eps

    def __call__(self, x: torch.Tensor):
        r = np.random.rand(1)
        if r < self.prob:
            c, h, w = x.shape[-3:]

            # sample index
            image_size = h
            h_index = random.randrange(-self.max_index, self.max_index)
            w_index = random.randrange(-self.max_index, self.max_index)

            # get l2 normalized Fourier basis
            fourier_basis = pytorch_fourier_analysis.fourier.basis.fourier_basis(
                image_size, h_index, w_index
            )  # [1, h, w]
            fourier_basis *= self.max_eps  # scale
            fourier_basis = fourier_basis.repeat(3, 1, 1)

            # random scaling
            fourier_basis[0, :, :] *= random.uniform(-1, 1)
            fourier_basis[1, :, :] *= random.uniform(-1, 1)
            fourier_basis[2, :, :] *= random.uniform(-1, 1)

            return torch.clamp(x + fourier_basis, 0.0, 1.0)
        else:
            return x


class PatchFourier(NoiseAugmentationBase):
    def __init__(
        self,
        prob: float,
        patch_size: int,
        randomize_patch_size: bool,
        max_index: int,
        max_eps: float,
    ):
        """
        Args
            prob: Probability of using Patch Gaussian
            patch_size: Size of patch. In the original paper, 25 for CIFAR-10 and 224 for ImageNet
            randomize_patch_size: Randomizing patch size or not
            max_index: Maximun index of Fourier basis
            max_eps: Maximun noise size in l2 norm
        """
        self.prob = prob
        self.patch_size = patch_size
        self.randomize_patch_size = randomize_patch_size
        self.max_index = max_index
        self.max_eps = max_eps

    def __call__(self, x: torch.Tensor):
        r = np.random.rand(1)
        if r < self.prob:
            c, w, h = x.shape[-3:]
            # sample index
            image_size = h
            h_index = random.randrange(-self.max_index, self.max_index)
            w_index = random.randrange(-self.max_index, self.max_index)

            # get l2 normalized Fourier basis
            fourier_basis = pytorch_fourier_analysis.fourier.basis.fourier_basis(
                image_size, h_index, w_index
            )  # [1, h, w]
            fourier_basis *= self.max_eps  # scale
            fourier_basis = fourier_basis.repeat(3, 1, 1)

            # random scaling
            fourier_basis[0, :, :] *= random.uniform(-1, 1)
            fourier_basis[1, :, :] *= random.uniform(-1, 1)
            fourier_basis[2, :, :] *= random.uniform(-1, 1)

            # generate mask
            patch_size = (
                random.randrange(1, self.patch_size + 1)
                if self.randomize_patch_size
                else self.patch_size
            )
            mask = self.sample_mask(x.size(), patch_size)

            return self.add_masked_noise(x, fourier_basis, mask)
        else:
            return x
