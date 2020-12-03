import random
from typing import Literal, Tuple, Union

import torch
import torch.fft
from torch.nn.modules.loss import _Loss
from torch.types import _device

from .attack import AttackWrapper


def logit_to_index(
    logit_h: torch.FloatTensor,
    logit_w: torch.FloatTensor,
    tau: float = 1.0,
    scale_logit: float = 5.0,
) -> torch.FloatTensor:
    """
    convert logit to index using gumbel softmax.
    the shape of returned index is (B,3,H,W).

    Args
        logit_h: logit of fourier basis about hight. its shape should be (B,3,H).
        logit_h: logit of fourier basis about width. its shape should be (B,3,W) or (B,3,W//2+1)).
        tau: tempalature of Gumbel softmax.
        scale_logit: scale factor of logit. NOTE: if this value is too small, Gumbel softmax does not work instead of thevalue of tau.
    """
    # apply gumbel softmax with "straight-through" trick.
    index_h_onehot = torch.nn.functional.gumbel_softmax(
        logit_h * scale_logit, tau=tau, hard=True
    )
    index_w_onehot = torch.nn.functional.gumbel_softmax(
        logit_w * scale_logit, tau=tau, hard=True
    )

    return torch.matmul(
        index_h_onehot.unsqueeze(-1), index_w_onehot.unsqueeze(-2)
    )  # (B,3,H,W)


def index_to_basis(index: torch.FloatTensor) -> torch.FloatTensor:
    """
    convert index to Fourier basis by 2D FFT.
    in order to apply 2D FFT, dim argument of torch.fft.irfftn should be =(-2,-1).

    Args
        index: its shape should be (B,3,H,W//2+1).
    """
    _, _, h, _ = index.size()
    basis = torch.fft.irfftn(index, s=(h, h), dim=(-2, -1))
    return basis / basis.norm(dim=(-2, -1))[:, :, None, None]


norm_type = Literal["linf"]


class FourierAttack(AttackWrapper):
    def __init__(
        self,
        input_size: int,
        mean: Tuple[float],
        std: Tuple[float],
        num_iteration: int,
        norm: norm_type,
        scale_eps: bool,
        scale_each: bool,
        avoid_target: bool,
        criterion: _Loss,
        device: Union[_device, str, None],
        scale_logit=10.0,
    ):
        super().__init__(input_size=input_size, mean=mean, std=std, device=device)
        self.num_iteration = num_iteration
        self.norm = norm
        self.criterion = criterion
        self.scale_logit = scale_logit

    def _forward(
        self,
        pixel_model: torch.nn.Module,
        pixel_x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return perturbed input in pixel space [0,255]
        """
        # if scale_eps is True, change eps adaptively.
        # this usually improve robustness against wide range of attack
        if self.scale_eps:
            if self.scale_each:
                rand = torch.rand(pixel_x.size(0), device=self.device)  # (B)
            else:
                rand = random.random() * torch.ones(
                    pixel_x.size(0), device=self.device
                )
            base_eps = rand.mul(self.eps_max)  # (B)
            step_size = rand.mul(self.step_size)  # (B)
        else:
            base_eps = self.eps_max * torch.ones(
                pixel_x.size(0), device=self.device
            )  # (B)
            step_size = self.step_size * torch.ones(
                pixel_x.size(0), device=self.device
            )  # (B)

        # init delta (=fourier basis)
        pixel_input = pixel_x.detach()
        pixel_input.requires_grad_()
        logit_h, logit_w = init_logit()

        # compute basis in pixel space
        if self.num_iteration:  # run iteration
            pixel_basis = self._run(pixel_model, pixel_input, logit_h, logit_w, target, base_eps, step_size)
        else:  # if self.num_iteration is 0, return just initialization result
            index = logit_to_index(logit_h, logit_w, scale_logit=self.scale_logit)
            pixel_basis = base_eps[:, None, None, None] * index_to_basis(index)

            pixel_basis.data = torch.clamp(pixel_input.data + pixel_basis.data, 0.0, 255.0) - pixel_input.data

        # NOTE: this return is in PIXEL SPACE (=[0,255])
        return pixel_input + pixel_basis

    def _run(
        self,
        pixel_model: torch.nn.Module,
        pixel_input: torch.FloatTensor,
        logit_h: torch.FloatTensor,
        logit_w: torch.FloatTensor,
        target: torch.Tensor,
        eps: torch.Tensor,
        step_size: torch.FloatTensor,
        scale_logit: float
    ) -> torch.FloatTensor:
        index = logit_to_index(logit_h, logit_w, scale_logit)
        pixel_basis = eps[:, None, None, None] * index_to_basis(index)

        logit = pixel_model(pixel_input + pixel_basis)

        for it in range(self.num_iteration):
            loss = self.criterion(logit, target)
            loss.backward()

            if self.avoid_target:
                grad_h = logit_h.grad.data  # to avoid target, increase the loss
                grad_w = logit_w.grad.data  # to avoid target, increase the loss
            else:
                grad_h = -logit_h.grad.data  # to hit target, decrease the loss
                grad_w = -logit_w.grad.data  # to hit target, decrease the loss

            if self.norm == "linf":
                grad_sign_h = grad_h.sign()
                grad_sign_w = grad_w.sign()

                logit_h.data = logit_h.data + step_size[:, None, None] * grad_sign_h
                logit_w.data = logit_w.data + step_size[:, None, None] * grad_sign_w

                # normalize logits
                logit_h.data = torch.clamp(logit_h.data, 0.0, 1.0)
                logit_w.data = torch.clamp(logit_w.data, 0.0, 1.0)

                # update basis
                index.data = logit_to_index(logit_h.data, logit_w.data, scale_logit)
                pixel_basis.data = eps[:, None, None, None] * index_to_basis(index.data)
                pixel_basis.data = torch.clamp(pixel_input.data + pixel_basis.data, 0.0, 255.0) - pixel_input.data
            else:
                raise NotImplementedError

            if it != self.num_iteration - 1:  # final iterarion
                logit = pixel_model(pixel_input + pixel_basis)
                pixel_basis.grad.data.zero_()

        return pixel_basis
