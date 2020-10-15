import pytest
import torchvision

import pytorch_fourier_analysis.shared


@pytest.fixture(scope="function")
def cifar10():
    transform = pytorch_fourier_analysis.shared.get_transform(
        input_size=32,
        mean=[0.49139968, 0.48215841, 0.44653091],
        std=[0.24703223, 0.24348513, 0.26158784],
        train=False,
        normalize=False,
    )
    dataset = torchvision.datasets.CIFAR10(
        root="data/cifar10", train=False, download=False, transform=transform
    )
    return dataset
