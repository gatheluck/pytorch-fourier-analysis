from typing import Optional

import torch
import torch.fft
import torchvision
import numpy as np

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
    if bandwidth % 2 == 0:
        raise ValueError

    # fft
    w = torch.fft.fftn(x, dim=(-2, -1))

    # shift
    if filter_mode == "high_pass":
        pass
    elif filter_mode == "low_pass":
        w = fft_shift(w)  # make low freq center
    else:
        raise NotImplementedError

    # apply bandpass filter
    mask = torch.zeros((x.size(-2), x.size(-1)), dtype=torch.bool, device=x.device)
    # if x.is_complex():
    #     mask = mask.unsqueeze(-1).repeat(1, 1, 2)
    center_h = x.size(-2) // 2
    center_w = x.size(-1) // 2

    if bandwidth == 1:
        mask[center_h, center_w] = True
    elif bandwidth > 1:
        halfband = bandwidth // 2
        mask[center_h - halfband: center_h + halfband, center_w - halfband: center_w + halfband] = True

    # if bandwidth > 0:
    #     halfband_l = np.ceil((x.size(-1) - bandwidth) / 2)
    #     halfband_s = np.floor((x.size(-1) - bandwidth) / 2)

    #     mask[0:(halfband_l + 1), :] = False
    #     mask[:, 0: (halfband_l + 1)] = False
    #     if halfband_s > 0:
    #         mask[-halfband_s:, :] = False
    #         mask[:, -halfband_s:] = False

        # mask[0:(halfband + 1), :] = False
        # mask[:, 0: (halfband + 1)] = False
        # if halfband > 0:
        #     mask[-halfband:, :] = False
        #     mask[:, -halfband:] = False

        # halfband = (x.size(-1) - bandwidth) // 2

        # mask[0:(halfband + 1), :] = False
        # mask[:, 0: (halfband + 1)] = False
        # if halfband > 0:
        #     mask[-halfband:, :] = False
        #     mask[:, -halfband:] = False

    w = w.where(mask, torch.zeros_like(w))

    # shift
    if filter_mode == "high_pass":
        pass
    elif filter_mode == "low_pass":
        w = ifft_shift(w)
    else:
        raise NotImplementedError

    # ifft
    x_filtered = torch.fft.ifftn(w, dim=(-2, -1))
    x_filtered = torch.real(x_filtered)

    # normalize
    if eps is None:
        return x_filtered
    else:
        # norms = x_filtered.view(x_filtered.size(0), -1).norm(dim=-1)  # (B)
        # return eps * (x_filtered / norms[:, None, None, None]), w.real

        norms_r = x_filtered[:, 0, :, :].view(x_filtered.size(0), -1).norm(dim=-1)  # (B)
        norms_g = x_filtered[:, 1, :, :].view(x_filtered.size(0), -1).norm(dim=-1)  # (B)
        norms_b = x_filtered[:, 2, :, :].view(x_filtered.size(0), -1).norm(dim=-1)  # (B)

        x_filtered[:, 0, :, :] /= norms_r[:, None, None]
        x_filtered[:, 1, :, :] /= norms_g[:, None, None]
        x_filtered[:, 2, :, :] /= norms_b[:, None, None]
        return eps * x_filtered, w.real

# def bandpass_filter(
#     x: torch.Tensor, bandwidth: int, filter_mode: str, eps: Optional[float] = None
# ) -> torch.Tensor:
#     """
#     Args
#         x: input tensor (image or noise). shape should be (B,C,H,W)
#         eps: size of x measured by l2 norm
#         bandwidth: the value should be in {0, h (or w)}
#         filter_mode: the value should be in {"high_pass", "low_pass"}
#     """

#     # fft
#     w = torch.fft.fftn(x, dim=(-2, -1))

#     # shift
#     if filter_mode == "high_pass":
#         pass
#     elif filter_mode == "low_pass":
#         w = fft_shift(w)  # make low freq center
#     else:
#         raise NotImplementedError

#     # apply bandpass filter
#     mask = torch.ones((x.size(-2), x.size(-1)), dtype=torch.bool, device=x.device)
#     if x.is_complex():
#         mask = mask.unsqueeze(-1).repeat(1, 1, 2)

#     if bandwidth > 0:
#         mask[0:bandwidth, 0:bandwidth] = False
#         mask[-bandwidth:-1, -bandwidth:-1] = False

#     w = w.where(mask, torch.zeros_like(w))

#     # shift
#     if filter_mode == "high_pass":
#         pass
#     elif filter_mode == "low_pass":
#         w = ifft_shift(w)
#     else:
#         raise NotImplementedError

#     # ifft
#     x_filtered = torch.fft.ifftn(w, dim=(-2, -1))
#     x_filtered = torch.real(x_filtered)

#     # normalize
#     if eps is None:
#         return x_filtered
#     else:
#         norms_r = x_filtered[:, 0, :, :].view(x_filtered.size(0), -1).norm(dim=-1)  # (B)
#         norms_g = x_filtered[:, 1, :, :].view(x_filtered.size(0), -1).norm(dim=-1)  # (B)
#         norms_b = x_filtered[:, 2, :, :].view(x_filtered.size(0), -1).norm(dim=-1)  # (B)

#         x_filtered[:, 0, :, :] /= norms_r[:, None, None]
#         x_filtered[:, 1, :, :] /= norms_g[:, None, None]
#         x_filtered[:, 2, :, :] /= norms_b[:, None, None]
#         # return eps * (x_filtered / norms[:, None, None, None])
#         return eps * x_filtered
