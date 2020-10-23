import random
from typing import Tuple, Union

import torch
from torch.nn.modules.loss import _Loss
from torch.types import _device

from .attack import AttackWrapper, normalized_random_init


class PgdAttack(AttackWrapper):
    def __init__(
        self,
        input_size: int,
        mean: Tuple[float],
        std: Tuple[float],
        num_iteration: int,
        eps_max: float,
        step_size: float,
        norm: str,
        rand_init: bool,
        scale_eps: bool,
        scale_each: bool,
        avoid_target: bool,
        criterion: _Loss,
        device: Union[_device, str, None],
    ):
        super().__init__(input_size=input_size, mean=mean, std=std, device=device)
        self.num_iteration = num_iteration
        self.eps_max = eps_max
        self.step_size = step_size
        self.norm = norm
        self.rand_init = rand_init
        self.scale_eps = scale_eps
        self.scale_each = scale_each
        self.avoid_target = avoid_target
        self.criterion = criterion

    def _forward(
        self, pixel_model: torch.nn.Module, pixel_x: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return perturbed input in pixel space [0,255]
        """
        # if scale_eps is True, change eps adaptively. this usually improve robustness against wide range of attack
        if self.scale_eps:
            if self.scale_each:
                rand = torch.rand(pixel_x.size()[0], device=self.device)  # (B)
            else:
                rand = random.random() * torch.ones(
                    pixel_x.size()[0], device=self.device
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

        # init delta
        pixel_input = pixel_x.detach()
        pixel_input.requires_grad_()
        pixel_delta = self._init_delta(pixel_input.size(), base_eps)  # (B,C,H,W)

        # compute delta in pixel space
        if self.num_iteration:  # run iteration
            pixel_delta = self._run(
                pixel_model, pixel_input, pixel_delta, target, base_eps, step_size,
            )
        else:  # if self.num_iteration is 0, return just initialization result
            pixel_delta.data = (
                torch.clamp(pixel_input.data + pixel_delta.data, 0.0, 255.0)
                - pixel_input.data
            )

        # IMPORTANT: this return is in PIXEL SPACE (=[0,255])
        return pixel_input + pixel_delta

    def _init_delta(self, shape: torch.Size, eps: torch.Tensor) -> torch.Tensor:
        """
        initialize delta. if self.rand_init is True, execute random initialization.
        """
        if self.rand_init:
            init_delta = normalized_random_init(
                shape, self.norm, device=self.device
            )  # initialize delta for linf or l2

            init_delta = eps[:, None, None, None] * init_delta  # scale by eps
            init_delta.requires_grad_()
            return init_delta
        else:
            return torch.zeros(shape, requires_grad=True, device=self.device)

    def _run(
        self,
        pixel_model: torch.nn.Module,
        pixel_input: torch.Tensor,
        pixel_delta: torch.Tensor,
        target: torch.Tensor,
        eps: torch.Tensor,
        step_size: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return perturbation (delta) in pixel_space [0,255]
        """
        logit = pixel_model(pixel_input + pixel_delta)
        if self.norm == "l2":
            l2_eps_max = eps

        for it in range(self.num_iteration):
            loss = self.criterion(logit, target)
            loss.backward()

            if self.avoid_target:
                grad = pixel_delta.grad.data  # to avoid target, increase the loss
            else:
                grad = -pixel_delta.grad.data  # to hit target, decrease the loss

            if self.norm == "linf":
                grad_sign = grad.sign()
                pixel_delta.data = (
                    pixel_delta.data + step_size[:, None, None, None] * grad_sign
                )
                pixel_delta.data = torch.max(
                    torch.min(pixel_delta.data, eps[:, None, None, None]),
                    -eps[:, None, None, None],
                )  # scale in [-eps, +eps]
                pixel_delta.data = (
                    torch.clamp(pixel_input.data + pixel_delta.data, 0.0, 255.0)
                    - pixel_input.data
                )
            elif self.norm == "l2":
                batch_size = pixel_delta.data.size(0)
                grad_norm = torch.norm(
                    grad.view(batch_size, -1), p=2.0, dim=1
                )  # IMPORTANT: if you set eps = 0.0 this leads nan
                normalized_grad = grad / grad_norm[:, None, None, None]
                pixel_delta.data = (
                    pixel_delta.data + step_size[:, None, None, None] * normalized_grad
                )
                l2_pixel_delta = torch.norm(
                    pixel_delta.data.view(batch_size, -1), p=2.0, dim=1
                )
                # check numerical instabitily
                proj_scale = torch.min(
                    torch.ones_like(l2_pixel_delta, device=self.device),
                    l2_eps_max / l2_pixel_delta,
                )
                pixel_delta.data = pixel_delta.data * proj_scale[:, None, None, None]
                pixel_delta.data = (
                    torch.clamp(pixel_input.data + pixel_delta.data, 0.0, 255.0)
                    - pixel_input.data
                )
            else:
                raise NotImplementedError

            if it != self.num_iteration - 1:
                logit = pixel_model(pixel_input + pixel_delta)
                pixel_delta.grad.data.zero_()

        return pixel_delta
