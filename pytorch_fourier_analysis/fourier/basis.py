from typing import List

import numpy as np
import torch


def fourier_basis(image_size: int, h_index: int, w_index: int) -> torch.FloatTensor:
    """
    Return l2 normalized 2D Fourier basis whose shape is [1, h, w].
    If you don't need gradient, numpy is faster than torch.

    Args
        image_size: Size of image
        h_index: Index of Fourier basis about h dimension
        w_index: Index of Fourier basis about w dimension
    """
    h, w = image_size, image_size
    h_center_index = h // 2
    w_center_index = w // 2

    spectrum_matrix = torch.zeros(h, w)
    spectrum_matrix[h_center_index + h_index, w_center_index + w_index] = 1.0
    spectrum_matrix[h_center_index - h_index, w_center_index - w_index] = 1.0

    spectrum_matrix = spectrum_matrix.numpy()
    spectrum_matrix = np.fft.ifftshift(
        spectrum_matrix
    )  # swap qadrant (low-freq centered to high-freq centered)

    fourier_basis = torch.from_numpy(np.fft.ifft2(spectrum_matrix).real).float()
    fourier_basis /= fourier_basis.norm()
    return fourier_basis


def basis_grid(grid_size: int, image_size: int) -> List[torch.FloatTensor]:
    """
    Utility function for visualizing Fourier basis.
    Returned list should be saved like, torchvision.utils.save_image(list_bases, savepath, nrow=grid_size)

    Args
        grid_size: Size of grid
        image_size: Size of image
    """
    basis_list = list()
    begin = int(-np.floor(grid_size / 2))
    end = int(np.ceil(grid_size / 2))
    for i_h in range(begin, end):
        for i_w in range(begin, end):
            basis = fourier_basis(image_size, i_h, i_w).repeat(3, 1, 1)
            basis_list.append(basis + 0.5)

    return basis_list
