import os
import sys
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.types import _device
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# from shift import ifft_shift
import pytorch_fourier_analysis.fourier.shift


def soft_basis(
    image_size: int,
    index: torch.FloatTensor,
    sigma: torch.FloatTensor,
    device: str
) -> torch.FloatTensor:
    """
    Args
        image_size: Size of image
        index: This values should be indices of Fourier Bases. The shape should be [b, 2]. h in [-1.0, 1.0], w in [0.0, 1.0]
        simga: Marched sigma of Gaussian. If there values are too small, that leads gradient vanishing. The shape should be [b]. sigma in [0.0, 1.0]
    """
    spectrum_matrix = soft_spectrum_matrix(image_size, index, sigma, device)  # [b, h, w]
    spectrum_matrix_comp = spectrum_matrix.unsqueeze(-1).repeat(1, 1, 1, 2)  # [b, h, w, 2]

    spectrum_matrix_comp = pytorch_fourier_analysis.fourier.shift.ifft_shift(
        spectrum_matrix_comp
    )  # shift from low freq center to high freq center for ifft.
    return spectrum_matrix_comp.irfft(2, normalized=True, onesided=False), spectrum_matrix.detach()


def soft_spectrum_matrix(
    image_size: int, mu: torch.FloatTensor, sigma: torch.FloatTensor, device: str
) -> torch.FloatTensor:
    """
    Return soft spectrum for Fourier Basis which differenciable about mu and sigma
    This function approximate spectrum for Fourier Basis by Gaussian.

    Args
        image_size: Size of image
        mu: Batched mean of Gaussian. This values should be indices of Fourier Bases. The shape should be [b, 2]. h in [-1.0, 1.0], w in [0.0, 1.0]
        simga: Marched sigma of Gaussian. If there values are too small, that leads gradient vanishing. The shape should be [b]. sigma in [0.0, 1.0]
    """
    def _calc_gaussian(
        x: torch.FloatTensor, mu: torch.FloatTensor, sigma: torch.FloatTensor
    ):
        expornet = -1.0 * (
            (x - mu.view(-1, 1, 1)) ** 2 / (2.0 * (sigma.view(-1, 1, 1) ** 2))
        )
        return torch.exp(expornet)

    batch_size = sigma.size(0)
    mu = mu * (image_size // 2)  # scale mu
    sigma = sigma * image_size  # scale sigma
    begin = -int(np.ceil(image_size / 2))
    end = int(np.floor(image_size / 2))

    grid_h, grid_w = torch.meshgrid(
        torch.tensor([i for i in range(begin, end)], dtype=torch.float, device=device),
        torch.tensor([i for i in range(begin, end)], dtype=torch.float, device=device),
    )
    grid_h = grid_h.view(1, image_size, image_size).repeat(
        batch_size, 1, 1
    )  # [b, h, w]
    grid_w = grid_w.view(1, image_size, image_size).repeat(
        batch_size, 1, 1
    )  # [b, h, w]

    soft_indices_pos_h = _calc_gaussian(grid_h, mu[:, 0], sigma)
    soft_indices_pos_w = _calc_gaussian(grid_w, mu[:, 1], sigma)
    soft_indices_neg_h = _calc_gaussian(grid_h, -mu[:, 0], sigma)
    soft_indices_neg_w = _calc_gaussian(grid_w, -mu[:, 1], sigma)

    soft_indices_pos = soft_indices_pos_h * soft_indices_pos_w
    soft_indices_neg = soft_indices_neg_h * soft_indices_neg_w

    soft_indices_pos = soft_indices_pos / soft_indices_pos.norm()
    soft_indices_neg = soft_indices_neg / soft_indices_neg.norm()

    return soft_indices_pos + soft_indices_neg


def create_soft_basis_grid(
    grid_size: int, image_size: int, saveroot: str, device: str
) -> None:
    """
    This function is utility function for visualizing Fourier basis.
    """
    list_bases = list()
    list_spectrum = list()
    begin = int(-np.floor(grid_size / 2))
    end = int(np.ceil(grid_size / 2))
    for i_h in range(begin, end):
        for i_w in range(begin, end):
            print("{}, {}".format(i_h, i_w))
            scaler = np.ceil(grid_size / 2)
            index = torch.tensor([i_h / scaler, i_w / scaler], dtype=torch.float, device=device, requires_grad=True).view(1, 2)
            sigma = torch.tensor([0.03], dtype=torch.float, device=device, requires_grad=True)
            basis, spectrum = soft_basis(image_size, index, sigma, device)
            basis = (basis / (basis.abs().max() * 2))
            basis = basis + 0.5
            print("mean: {}".format(basis[0].mean()))
            print("max: {}".format(basis[0].max()))
            print("min: {}".format(basis[0].min()))
            print(" ")
            list_bases.append(basis.detach())
            list_spectrum.append(spectrum.detach())

    torchvision.utils.save_image(list_bases, os.path.join(saveroot, "soft_basis_grid.png"), nrow=grid_size)
    torchvision.utils.save_image(list_spectrum, os.path.join(saveroot, "soft_spectrum_grid.png"), nrow=grid_size)


if __name__ == "__main__":
    device = "cpu"
    mu = torch.tensor([[0.5, 0.0], [-1.0, 1.0]], dtype=torch.float, requires_grad=True)
    sigma = torch.tensor([0.1, 0.1], dtype=torch.float, requires_grad=True)
    print(mu.shape)
    print(sigma.shape)
    s = soft_spectrum_matrix(32, mu, sigma=sigma, device=device)
    print(s)
    print(s.shape)
    torchvision.utils.save_image(s[0], "../../logs/soft_spectrum_0.png")
    torchvision.utils.save_image(s[1], "../../logs/soft_spectrum_1.png")