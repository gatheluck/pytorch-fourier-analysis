import torch
import torchvision

import pytorch_fourier_analysis.shared
import pytorch_fourier_analysis.noiseaugments


class TestFourier:
    def test_generate_sample(self, cifar10):
        fourier_augment = pytorch_fourier_analysis.noiseaugments.Fourier(
            prob=1.0, max_index=15, max_eps=4.0
        )

        dataset = cifar10
        dataset.transform.transforms.append(
            fourier_augment
        )  # dataset.transform is torchvision.transforms.Compose object
        loader = torch.utils.data.DataLoader(dataset, 32, shuffle=False, num_workers=8)

        for x, _ in loader:
            torchvision.utils.save_image(x, "logs/fourier.png")
            break


class TestPatchFourier:
    def test_generate_sample(self, cifar10):
        patch_fourier_augment = pytorch_fourier_analysis.noiseaugments.PatchFourier(
            prob=1.0,
            patch_size=25,
            randomize_patch_size=True,
            max_index=15,
            max_eps=4.0,
        )

        dataset = cifar10
        dataset.transform.transforms.append(
            patch_fourier_augment
        )  # dataset.transform is torchvision.transforms.Compose object
        loader = torch.utils.data.DataLoader(dataset, 32, shuffle=False, num_workers=8)

        for x, _ in loader:
            torchvision.utils.save_image(x, "logs/patch_fourier.png")
            break
