import torch
import math


# def fft_shift(x: torch.Tensor) -> torch.Tensor:
#     r"""
#     PyTorch version of np.fftshift.

#     Args
#         x: Input tensor in image space. Its shape should be [(B), C, H, W, 2]
#     """
#     dims = [i for i in range(1 if x.dim() == 4 else 2, x.dim() - 1)]  # [H, W]
#     shift = [x.size(dim) // 2 for dim in dims]
#     return torch.roll(x, shift, dims)


# def ifft_shift(x: torch.Tensor) -> torch.Tensor:
#     r"""
#     PyTorch version of np.ifftshift.

#     Args
#         x: Input tensor in Fourier space. Its shape should be [(B), C, H, W, 2]
#     """
#     dims = [i for i in range(x.dim() - 2, 0 if x.dim() == 4 else 1, -1)]  # [H, W]
#     shift = [int(math.ceil(x.size(dim) / 2)) for dim in dims]
#     return torch.roll(x, shift, dims)


def fft_shift(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch version of np.fftshift.

    Args
        x: Input tensor in image space (it might be complex tensor). Its shape should be [(B), C, H, W, (2)]
    """
    h_dim = -2 if x.is_complex() else -3
    w_dim = -1 if x.is_complex() else - 2
    dims = [h_dim, w_dim]
    shift = [x.size(dim) // 2 for dim in dims]
    return torch.roll(x, shift, dims)


def ifft_shift(w: torch.Tensor) -> torch.Tensor:
    h_dim = -2 if w.is_complex() else -3
    w_dim = -1 if w.is_complex() else - 2
    dims = [w_dim, h_dim]
    shift = [int(math.ceil(w.size(dim) / 2)) for dim in dims]
    return torch.roll(w, shift, dims)
