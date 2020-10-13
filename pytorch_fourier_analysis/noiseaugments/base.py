import math
import random

import numpy as np
import torch


class NoiseAugmentationBase:
    def __init__(self):
        pass

    def add_masked_noise(self, x: torch.Tensor, noise: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        noise_x = torch.clamp(x + noise, 0.0, 1.0)
        masked_noise_x = torch.where(mask, noise_x, x)  # apply noise only masked region
        return torch.clamp(masked_noise_x, 0.0, 1.0)

    def sample_mask(self, input_size: torch.Size, patch_size: int) -> torch.BoolTensor:
        """
        Args:
            size: Size of input tensor.
            patch_size: Size of patch including mask. If -1, return full size patch.
        """
        h, w = input_size[-2], input_size[-1]

        # if window_size == -1, return all True mask.
        if patch_size == -1:
            return torch.ones(input_size, dtype=torch.bool)

        # sample window center. if window size is odd, sample from pixel position. if even, sample from grid position.
        h_center = random.randrange(0, h)
        w_center = random.randrange(0, w)

        h_begin = np.clip((h_center - math.floor(patch_size / 2)), 0, h)
        w_begin = np.clip((w_center - math.floor(patch_size / 2)), 0, w)
        h_end = np.clip((h_center + math.ceil(patch_size / 2)), 0, h)
        w_end = np.clip((w_center + math.ceil(patch_size / 2)), 0, w)

        mask = torch.zeros(input_size, dtype=torch.bool)  # all elements are False
        mask[..., h_begin:h_end, w_begin:w_end] = torch.ones(
            input_size, dtype=torch.bool
        )[..., h_begin:h_end, w_begin:w_end]
        return mask
