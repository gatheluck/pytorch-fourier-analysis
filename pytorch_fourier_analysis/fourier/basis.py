import torch
import torchvision
import numpy as np
from typing import List, Tuple, Union
from torch.types import _device

from shift import ifft_shift


def basis(
    size: Union[List[int], Tuple[int, ...]],
    indices: torch.Tensor,
    device: Union[_device, str, None],
) -> torch.Tensor:
    """

    Args
        size:
        indices: Batched index (i, j) of Fourier basis. Shape should be [b, 2]
        device:
        requires_grad: 
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
    fourier_space_tensor = ifft_shift(fourier_space_tensor)
    return fourier_space_tensor.irfft(2, normalized=True, onesided=False)


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


def soft_spectrum_matrix(
    image_size: int, mu: torch.FloatTensor, sigma: torch.FloatTensor
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
        torch.tensor([i for i in range(begin, end)], dtype=torch.float),
        torch.tensor([i for i in range(begin, end)], dtype=torch.float),
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

    return soft_indices_pos + soft_indices_neg


def create_fourier_basis_grid(
    grid_size: int, image_size: int, savepath: str, device: str
) -> None:
    """
    This function is utility function for visualizing Fourier basis.
    """
    list_bases = list()
    begin = int(-np.ceil(grid_size / 2))
    end = int(np.ceil(grid_size / 2))
    for i_h in range(begin, end):
        for i_w in range(begin, end):
            indices = torch.tensor([i_h, i_w], dtype=torch.int).view(1, 2).cuda()
            list_bases.append(
                basis((1, image_size, image_size), indices, device).repeat(3, 1, 1)
                + 0.5
            )

    torchvision.utils.save_image(list_bases, savepath, nrow=grid_size)


if __name__ == "__main__":
    # create_fourier_basis_grid(10, 24, savepath="../../logs/basis.png", device="cuda")

    # indices = torch.tensor([0, 0], dtype=torch.int, requires_grad=True).view(1, 2).cuda()
    # fourier_basis = basis((1, 32, 32), indices, "cuda").repeat(3, 1, 1)
    # print(fourier_basis)

    mu = torch.tensor([[0.5, 0.0], [-1.0, 1.0]], dtype=torch.float, requires_grad=True)
    sigma = torch.tensor([0.1, 0.1], dtype=torch.float, requires_grad=True)
    print(mu.shape)
    print(sigma.shape)
    s = soft_spectrum_matrix(32, mu, sigma=sigma)
    print(s)
    print(s.shape)
    torchvision.utils.save_image(s[0], "../../logs/soft_spectrum_0.png")
    torchvision.utils.save_image(s[1], "../../logs/soft_spectrum_1.png")
