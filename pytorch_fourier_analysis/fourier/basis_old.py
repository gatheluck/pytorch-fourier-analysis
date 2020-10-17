import torch
import torchvision
import numpy as np
from typing import List, Tuple, Union
from torch.types import _device

import pytorch_fourier_analysis.fourier.shift


def basis(
    size: Union[List[int], Tuple[int, ...]],
    indices: torch.Tensor,
    device: Union[_device, str, None],
) -> torch.Tensor:
    """
    Return Fourier undifferentiable Fourier basis. Shape is [b, h, w].

    Args
        size:
        indices: Batched index (i, j) of Fourier basis. Shape should be [b, 2]
        device:
    """
    b, h, w = size[0], size[-2], size[-1]

    # create 2d mesh grid
    grid_h, grid_w = zero_centered_meshgrid_2d(h, w, device=device, is_symmetric=False)
    grid = torch.stack([grid_h, grid_w], dim=-1).repeat(b, 1, 1, 1)  # [b, h, w, 2]
    indices_hw = indices.view(b, 1, 1, 2).repeat(1, h, w, 1).to(device)

    ones = torch.ones((b, h, w), dtype=torch.float, device=device)
    zeros = torch.zeros((b, h, w), dtype=torch.float, device=device)

    # fourier basis has non zero element at (i,j) and (-i,-j).
    condition_one = torch.logical_and(
        grid[..., 0] == indices_hw[..., 0], grid[..., 1] == indices_hw[..., 1]
    )
    condition_two = torch.logical_and(
        -grid[..., 0] == indices_hw[..., 0], -grid[..., 1] == indices_hw[..., 1]
    )
    fourier_space_tensor = torch.where(
        torch.logical_or(condition_one, condition_two), ones, zeros
    )

    fourier_space_tensor = fourier_space_tensor.view(b, h, w, 1).repeat(
        1, 1, 1, 2
    )  # [b, h, w, 2]
    # shift from low freq center to high freq center for ifft.
    fourier_space_tensor = pytorch_fourier_analysis.fourier.shift.ifft_shift(fourier_space_tensor)
    return fourier_space_tensor.irfft(2, normalized=True, onesided=False)


def basis_numpy(size: Union[List[int], Tuple[int, ...]], h_index: int, w_index: int):
    """

    
    """
    h, w = size[-2], size[-1]
    h_center_index = h // 2
    w_center_index = w // 2

    spectrum_matrix = torch.zeros(h, w)
    spectrum_matrix[h_center_index + h_index, w_center_index + w_index] = 1.0
    spectrum_matrix[h_center_index - h_index, w_center_index - w_index] = 1.0

    spectrum_matrix = spectrum_matrix.numpy()
    spectrum_matrix = np.fft.ifftshift(spectrum_matrix)  # swap qadrant (low-freq centered to high-freq centered)

    fourier_base = torch.from_numpy(np.fft.ifft2(spectrum_matrix).real).float()
    fourier_base /= fourier_base.norm()
    return fourier_base


def zero_centered_meshgrid_2d(
    h: int, w: int, is_symmetric: bool, device: Union[_device, str, None]
) -> Tuple[torch.Tensor, ...]:
    r"""
    Returns zero centered 2D symmetric meshgrid.

    Args:
        h: hight of meshgrid
        w: width of meshgrid
        is_symmetric: if True, return becomes symmetric
        device: the desired device of returned tensor.

    """
    grid_h, grid_w = torch.meshgrid(
        torch.tensor([i for i in range(h)], device=device),
        torch.tensor([i for i in range(w)], device=device),
    )

    zero_centered_grid_h = grid_h - ((h - 1) / 2.0)
    zero_centered_grid_w = grid_w - ((w - 1) / 2.0)

    if is_symmetric:
        zero_centered_grid_h = (
            (zero_centered_grid_h.ceil() + zero_centered_grid_h.floor()) / 2.0
        ).int()
        zero_centered_grid_w = (
            (zero_centered_grid_w + zero_centered_grid_w) / 2.0
        ).int()
    else:
        zero_centered_grid_h = zero_centered_grid_h.floor().int()
        zero_centered_grid_w = zero_centered_grid_w.floor().int()

    return zero_centered_grid_h, zero_centered_grid_w


def create_fourier_basis_grid(
    grid_size: int, image_size: int, savepath: str, device: str
) -> None:
    """
    This function is utility function for visualizing Fourier basis.
    """
    list_bases = list()
    begin = int(-np.floor(grid_size / 2))
    end = int(np.ceil(grid_size / 2))
    for i_h in range(begin, end):
        for i_w in range(begin, end):
            indices = torch.tensor([i_h, i_w], dtype=torch.int).view(1, 2).cuda()
            list_bases.append(
                basis((1, image_size, image_size), indices, device).repeat(3, 1, 1)
                + 0.5
            )

    torchvision.utils.save_image(list_bases, savepath, nrow=grid_size)


def create_fourier_basis_grid_np(
    grid_size: int, image_size: int, savepath: str
) -> None:
    """
    This function is utility function for visualizing Fourier basis.
    """
    list_bases = list()
    begin = int(-np.floor(grid_size / 2))
    end = int(np.ceil(grid_size / 2))
    for i_h in range(begin, end):
        for i_w in range(begin, end):
            basis = basis_numpy([image_size, image_size], i_h, i_w).repeat(3, 1, 1)
            list_bases.append(basis + 0.5)

    torchvision.utils.save_image(list_bases, savepath, nrow=grid_size)


if __name__ == "__main__":
    pass
    # create_fourier_basis_grid(10, 24, savepath="../../logs/basis.png", device="cuda")

    # indices = torch.tensor([0, 0], dtype=torch.int, requires_grad=True).view(1, 2).cuda()
    # fourier_basis = basis((1, 32, 32), indices, "cuda").repeat(3, 1, 1)
    # print(fourier_basis)

    # mu = torch.tensor([[0.5, 0.0], [-1.0, 1.0]], dtype=torch.float, requires_grad=True)
    # sigma = torch.tensor([0.1, 0.1], dtype=torch.float, requires_grad=True)
    # print(mu.shape)
    # print(sigma.shape)
    # s = soft_spectrum_matrix(32, mu, sigma=sigma)
    # print(s)
    # print(s.shape)
    # torchvision.utils.save_image(s[0], "../../logs/soft_spectrum_0.png")
    # torchvision.utils.save_image(s[1], "../../logs/soft_spectrum_1.png")