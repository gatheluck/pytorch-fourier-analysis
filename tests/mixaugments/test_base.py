import os
import sys

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import mixaugments


class TestMixAugmentationBase:
    def test_calc_mixed_loss(self):
        output = torch.randn(16, 10)
        t = torch.randint(low=0, high=10, size=(16,), dtype=torch.long)
        rand_index = torch.randperm(t.size(0))
        criterion = torch.nn.CrossEntropyLoss()

        ce_loss_a = criterion(output, t)
        ce_loss_b = criterion(output, t[rand_index])
        assert (
            mixaugments.MixAugmentationBase()
            ._calc_mixed_loss(output, t, rand_index, 1.0, criterion)
            .equal(ce_loss_a)
        )
        assert (
            mixaugments.MixAugmentationBase()
            ._calc_mixed_loss(output, t, rand_index, 0.0, criterion)
            .equal(ce_loss_b)
        )
        assert (
            mixaugments.MixAugmentationBase()
            ._calc_mixed_loss(output, t, rand_index, 0.5, criterion)
            .equal((0.5 * ce_loss_a) + (0.5 * ce_loss_b))
        )

    # test data for test_sample_mask
    params = {
        "full size mask": (
            torch.Size([3, 32, 32]),
            -1,
            torch.ones(3, 32, 32, dtype=torch.bool),
        )
    }

    @pytest.mark.parametrize(
        "input_size, mask_size, expect", list(params.values()), ids=list(params.keys())
    )
    def test_sample_mask(self, input_size, mask_size, expect):
        assert (
            mixaugments.MixAugmentationBase()
            ._sample_mask(input_size, mask_size)
            .equal(expect)
        )
