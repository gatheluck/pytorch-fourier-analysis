import torch
import numpy as np
from typing import List, Tuple, Union
from torch.types import _device

from shift import ifft_shift


def basis(
    size: Union[List[int], Tuple[int, ...]],
    indices: torch.Tensor,
    device: Union[_device, str, None],
    requires_grad: bool,
) -> torch.Tensor:
    """
    """
    b, h, w = size[0], size[-2], size[-1]

    grid_h, grid_w = zero_centered_meshgrid_2d(h, w, device=device, is_symmetric=False)
    grid = torch.stack([grid_h, grid_w], dim=-1).repeat(b, 1, 1, 1)  # [b, h, w, 2]

    # if indices are specified, return basis are created from same indices.
    # if indices are not specified, return basis are created from uniformly sampled indices.
    # if indices:
    #     indices_h = torch.tensor([indices[0]], dtype=torch.int, device=device).repeat(b)
    #     indices_w = torch.tensor([indices[1]], dtype=torch.int, device=device).repeat(b)
    # else:
    #     indices_h = torch.randint(
    #         int(grid_h.min().item()),
    #         int(grid_h.max().item() + 1),
    #         size=(b,),
    #         dtype=torch.int,
    #         device=device,
    #     )
    #     indices_w = torch.randint(
    #         int(grid_w.min().item()),
    #         int(grid_w.max().item() + 1),
    #         size=(b,),
    #         dtype=torch.int,
    #         device=device,
    #     )

    # indices_hw = (
    #     torch.stack([indices_h, indices_w], dim=-1).view(b, 1, 1, 2).repeat(1, h, w, 1)
    # )  # [b, h, w, 2]
    indices_hw = indices.view(b, 1, 1, 2).repeat(1, h, w, 1)

    ones = torch.ones((b, h, w), dtype=torch.float, device=device)
    zeros = torch.zeros((b, h, w), dtype=torch.float, device=device)
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
    # fourier_space_tensor = torch.cat([fourier_space_tensor.view(b, h, w, 1), torch.zeros_like(fourier_space_tensor.view(b, h, w, 1))], dim=-1)

    # fourier_space_tensor_numpy = fourier_space_tensor.cpu().numpy()

    # print(indices)
    fourier_space_tensor = ifft_shift(fourier_space_tensor)
    # fourier_space_tensor_numpy = np.fft.ifftshift(fourier_space_tensor_numpy)
    # print(fourier_space_tensor[..., 0])
    # print(torch.from_numpy(fourier_space_tensor_numpy)[..., 0])

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


if __name__ == "__main__":
    import torchvision
    # print(symmetric_2d_meshgrid(7,7,'cuda'))
    # basis((3, 6, 6), None, "cuda", True)

    list_bases = list()
    for i_h in range(-5, 5):
        for i_w in range(-5, 5):
            indices = torch.tensor([i_h, i_w], dtype=torch.int).view(1, 2).cuda()
            list_bases.append(basis((1, 12, 12), indices, "cuda", True).repeat(3, 1, 1) + 0.5)

    torchvision.utils.save_image(list_bases, '../../logs/basis.png', nrow=10)
