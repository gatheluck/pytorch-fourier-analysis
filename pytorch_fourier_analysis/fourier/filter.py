from typing import Optional

import torch
import torch.fft
import torchvision

from .shift import fft_shift, ifft_shift


def bandpass_filter(
    x: torch.Tensor, bandwidth: int, filter_mode: str, eps: Optional[float] = None
) -> torch.Tensor:
    """
    Args
        x: input tensor (image or noise). shape should be (B,C,H,W)
        eps: size of x measured by l2 norm
        bandwidth: the value should be in {0, h (or w)}
        filter_mode: the value should be in {"high_pass", "low_pass"}
    """

    # fft
    w = torch.fft.fftn(x, dim=(-2, -1))

    # shift
    if filter_mode == "high_pass":
        w = fft_shift(w)  # make low freq center
    elif filter_mode == "low_pass":
        pass
    else:
        raise NotImplementedError

    # apply bandpass filter
    mask = torch.zeros((x.size(-2), x.size(-1)), dtype=torch.bool, device=x.device)
    if x.is_complex():
        mask = mask.unsqueeze(-1).repeat(1, 1, 2)

    if bandwidth > 0:
        mask[0:bandwidth, 0:bandwidth] = True
        mask[-bandwidth:-1, -bandwidth:-1] = True

    w = w.where(mask, torch.zeros_like(w))

    # shift
    if filter_mode == "high_pass":
        w = ifft_shift(w)
    elif filter_mode == "low_pass":
        pass
    else:
        raise NotImplementedError

    # ifft
    x_filtered = torch.fft.ifftn(w, dim=(-2, -1))
    x_filtered = torch.real(x_filtered)

    # normalize
    if eps is None:
        return x_filtered
    else:
        norms = x_filtered.view(x_filtered.size(0), -1).norm(dim=-1)  # (B)
        return eps * (x_filtered / norms[:, None, None, None])
