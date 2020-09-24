import torch
from typing import List, Tuple, Union


def basis(size: Union[List[int], Tuple[int, ...]], indices: Union[List[int], Tuple[int, ...], None], device: Union[int, str, None], requires_grad: bool) -> torch.Tensor:
    """
    """
    fourier_space_tensor = torch.zeros(size=size, dtype=torch.float, device=device, requires_grad=requires_grad)

    h = torch.tensor([i for i in range(size[-2])])
    w = torch.tensor([i for i in range(size[-1])])

    grid_h, grid_w = torch.meshgrid(h, w)

    grid_h_zero_centered = grid_h - ((size[-2] - 1) / 2.0)
    grid_w_zero_centered = grid_w - ((size[-1] - 1) / 2.0)

    grid_h_zero_centered = ((grid_h_zero_centered.ceil() + grid_h_zero_centered.floor()) / 2.0).int()
    grid_w_zero_centered = ((grid_w_zero_centered.ceil() + grid_w_zero_centered.floor()) / 2.0).int()

    print(grid_h_zero_centered)
    print(grid_w_zero_centered)

    grid_zero_centered = torch.stack([grid_h_zero_centered, grid_w_zero_centered], dim=-1)  # [h, w, 2]
    # print(grid_zero_centered)

    # wave_indices = 


def symmetric_2d_meshgrid(h: int, w: int, device: Union[int, str, None]) -> Tuple[torch.Tensor, ...]:
    grid_h, grid_w = torch.meshgrid(torch.tensor([i for i in range(h)], device=device), torch.tensor([i for i in range(w)], device=device))

    grid_h_symmetric = grid_h - ((h - 1) / 2.0)
    grid_w_symmetric = grid_w - ((w - 1) / 2.0)

    grid_h_symmetric = ((grid_h_symmetric.ceil() + grid_h_symmetric.floor()) / 2.0).int()
    grid_w_symmetric = ((grid_w_symmetric.ceil() + grid_w_symmetric.floor()) / 2.0).int()

    return grid_h_symmetric, grid_w_symmetric





if __name__ == '__main__':
    print(symmetric_2d_meshgrid(7,7,'cuda'))