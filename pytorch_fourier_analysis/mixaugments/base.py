import math
import random

import numpy as np
import torch


class MixAugmentationBase:
    def __init__(self):
        pass

    def _calc_mixed_loss(
        self,
        output: torch.tensor,
        t: torch.tensor,
        rand_index: torch.tensor,
        lam: float,
        criterion: torch.nn.modules.loss._Loss,
    ) -> torch.tensor:
        """
        Calcurate loss for mix augmentation.
        lamda * (loss for t_a)   + (1.0 - lamda) * (loss for t_b)

        Args
            output: Output logits from model. Shape should be [b, num_classes].
            t: Target classes. Shape should be [b].
            rand_index: Index of random swaping. Shape should be[b].
            lam: Weight of "a" side target. This value is same between same batch.
            criterion: Function which map from logits to loss.
        """
        t_a, t_b = t, t[rand_index]
        loss = (lam * criterion(output, t_a)) + ((1.0 - lam) * criterion(output, t_b))
        return loss

    def _sample_mask(self, input_size: torch.Size, mask_size: int) -> torch.BoolTensor:
        """
        Args:
            size: Size of input tensor.
            mask_size: Size of mask. If -1, return full size mask.
        """
        h, w = input_size[-2], input_size[-1]

        # if window_size == -1, return all True mask.
        if mask_size == -1:
            return torch.ones(input_size, dtype=torch.bool)

        # sample window center. if window size is odd, sample from pixel position. if even, sample from grid position.
        h_center = random.randrange(0, h)
        w_center = random.randrange(0, w)

        h_begin = np.clip((h_center - math.floor(mask_size / 2)), 0, h)
        w_begin = np.clip((w_center - math.floor(mask_size / 2)), 0, w)
        h_end = np.clip((h_center + math.ceil(mask_size / 2)), 0, h)
        w_end = np.clip((w_center + math.ceil(mask_size / 2)), 0, w)

        mask = torch.zeros(input_size, dtype=torch.bool)  # all elements are False
        mask[..., h_begin:h_end, w_begin:w_end] = torch.ones(
            input_size, dtype=torch.bool
        )[..., h_begin:h_end, w_begin:w_end]
        return mask
