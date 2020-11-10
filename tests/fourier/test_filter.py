import os
import sys
import random

import torch
import torchvision
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import fourier

if __name__ == "__main__":
    import numpy as np

    grid_size = 31
    image_size = 31

    # w = torch.zeros((1, 3, 31, 31), dtype=torch.cfloat)
    # w[:, :, 16, 16] = torch.complex(torch.tensor([1.0]), torch.tensor([1.0]))
    # w[:, :, 14, 14] = torch.complex(torch.tensor([1.0]), torch.tensor([1.0]))

    # w = fourier.ifft_shift(w)
    # x = torch.fft.ifftn(w, dim=(-2, -1)).real + 0.5 

    # torchvision.utils.save_image(x, "logs/new_fft.png")

    basis_list = list()
    begin = int(-np.floor(grid_size / 2))
    end = int(np.ceil(grid_size / 2))
    for i_h in range(begin, end):
        for i_w in range(begin, end):
            height, width = image_size, image_size
            h_center_index = height // 2
            w_center_index = width // 2

            w = torch.zeros((3, height, width), dtype=torch.cfloat)
            w[:, h_center_index + i_h, w_center_index + i_h] = torch.complex(torch.tensor([1.0]), torch.tensor([1.0]))
            w[:, h_center_index - i_h, w_center_index - i_w] = torch.complex(torch.tensor([1.0]), torch.tensor([1.0]))

            # w = fourier.ifft_shift(w)

            w_np = w.real.numpy()
            w_np = np.fft.ifftshift(w_np)
            # fourier_basis = torch.fft.ifftn(w, dim=(-2, -1))
            fourier_basis = torch.from_numpy(np.fft.ifft2(w_np).real).float()

            # fourier_basis = torch.fft.ifftn(w, dim=(-2, -1))
            # fourier_basis = (fourier_basis.real + fourier_basis.imag) / 2.0
            fourier_basis[0, :, :] /= fourier_basis[0].norm()
            fourier_basis[1, :, :] /= fourier_basis[1].norm()
            fourier_basis[2, :, :] /= fourier_basis[2].norm()

            basis_list.append(fourier_basis + 0.5)

    torchvision.utils.save_image(basis_list, "logs/new_fft_list.png", nrow=grid_size)

    # x = Image.open("tests/testdata/blackswan.png")
    # transform = torchvision.transforms.Compose(
    #     [torchvision.transforms.CenterCrop(32), torchvision.transforms.ToTensor()]
    # )
    # x = transform(x).unsqueeze(0)  # (B,C,H,W)

    # scale = random.uniform(0, 1)
    # gaussian = torch.normal(mean=0.0, std=scale, size=list(x.size()))

    # # x = fourier.bandpass_filter(x, bandwidth=224, filter_mode="low_pass")
    # gaussian_filtered = fourier.bandpass_filter(gaussian, bandwidth=31, filter_mode="low_pass")
    # cated = torch.cat([gaussian, gaussian_filtered]) + 0.5
    # cated = torch.clamp(cated, 0, 1.0)
    # print(cated.shape)

    # torchvision.utils.save_image(cated, "logs/bandpass.png")
