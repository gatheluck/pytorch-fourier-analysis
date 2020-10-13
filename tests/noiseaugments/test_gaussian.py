import os
import sys
import torch
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import noiseaugments
from pytorch_fourier_analysis import shared


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
