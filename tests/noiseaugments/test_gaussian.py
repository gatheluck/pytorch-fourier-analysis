import os
import random
import sys

import pytest
import torch
import torchvision
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import noiseaugments
from pytorch_fourier_analysis import shared


torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True


class TestGetGaussianNose:
    params = {
        "a": (None, None, False),
        "b": (None, None, True),
        "c": (None, 8, False),
        "d": (None, 8, True),
        "e": ("low_pass", None, False),
        "f": ("low_pass", None, True),
        "g": ("high_pass", None, False),
        "h": ("high_pass", None, True),
    }

    @pytest.mark.parametrize(
        "mode, bandwidth, adjust_eps",
        list(params.values()),
        ids=list(params.keys()),
    )
    def test_no_bandpass_filter(self, mode, bandwidth, adjust_eps):
        mean = 0.0
        std = random.uniform(0, 1)
        size = (3, 32, 32)

        torch.manual_seed(123)
        expect = torch.normal(mean=mean, std=std, size=size)  # (c,h,w)

        torch.manual_seed(123)
        assert noiseaugments.get_gaussian_noise(mean, std, size, mode, bandwidth, adjust_eps).equal(expect) 




if __name__ == "__main__":
    # test Gaussian
    transform = shared.get_transform(
        32,
        mean=[0.49139968, 0.48215841, 0.44653091],
        std=[0.24703223, 0.24348513, 0.26158784],
        train=False,
        normalize=False,
        optional_transform=[
            noiseaugments.Gaussian(prob=1.0, max_scale=1.0, randomize_scale=True)
        ],
    )
    dataset = torchvision.datasets.CIFAR10(
        root="../../data/cifar10", train=False, download=False, transform=transform
    )
    loader = torch.utils.data.DataLoader(dataset, 32, shuffle=False, num_workers=8)

    for x, _ in loader:
        torchvision.utils.save_image(x, "../../logs/gaussian.png")
        break

    # test PatchGaussian
    transform = shared.get_transform(
        32,
        mean=[0.49139968, 0.48215841, 0.44653091],
        std=[0.24703223, 0.24348513, 0.26158784],
        train=False,
        normalize=False,
        optional_transform=[
            noiseaugments.PatchGaussian(
                prob=1.0,
                patch_size=25,
                randomize_patch_size=True,
                max_scale=1.0,
                randomize_scale=True,
            )
        ],
    )
    dataset = torchvision.datasets.CIFAR10(
        root="../../data/cifar10", train=False, download=False, transform=transform
    )
    loader = torch.utils.data.DataLoader(dataset, 32, shuffle=False, num_workers=8)

    for x, _ in loader:
        torchvision.utils.save_image(x, "../../logs/patch_gaussian.png")
        break
