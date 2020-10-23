from typing import Tuple, Union

import torch
from torch.types import _device


def normalized_random_init(
    shape: torch.Size, norm: str, device: Union[_device, str, None]
) -> torch.Tensor:
    """
    Args:
        shape: shape of expected tensor. eg.) (B,C,H,W)
        norm: type of norm
    """
    if norm == "linf":
        init = (
            2.0 * torch.rand(shape, dtype=torch.float, device=device) - 1.0
        )  # values are in [-1, +1]
    elif norm == "l2":
        init = 2.0 * torch.randn(
            shape, dtype=torch.float, device=device
        )  # values in init are sampled form N(0,1)
        init_norm = torch.norm(init.view(init.size(0), -1), p=2.0, dim=1)  # (B)
        normalized_init = init / init_norm[:, None, None, None]

        dim = init.size(1) * init.size(2) * init.size(3)
        rand_norm = torch.pow(
            torch.rand(init.size(0), dtype=torch.float, device=device), 1.0 / dim
        )
        init = normalized_init * rand_norm[:, None, None, None]
    else:
        raise NotImplementedError

    return init


class Normalizer(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        mean: Tuple[float],
        std: Tuple[float],
        device: Union[_device, str, None],
        from_pixel_space: bool = True,
    ):
        """
        Differetiable normalizer. Input tensor might be in pixel space: [0, 255.0] or unit space: [0, 1.0]
        """
        super().__init__()
        self.from_pixel_space = from_pixel_space
        num_channel = len(mean)

        mean_list = [
            torch.full((input_size, input_size), mean[i], device=device)
            for i in range(num_channel)
        ]  # 3 x [h, w]
        self.mean = torch.unsqueeze(torch.stack(mean_list), 0)  # [1, 3, h, w]

        std_list = [
            torch.full((input_size, input_size), std[i], device=device)
            for i in range(num_channel)
        ]  # 3 x [h, w]
        self.std = torch.unsqueeze(torch.stack(std_list), 0)  # [1, 3, h, w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.from_pixel_space:
            x = x / 255.0
        return x.sub(self.mean).div(self.std)


class Denormalizer(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        mean: Tuple[float],
        std: Tuple[float],
        device: Union[_device, str, None],
        to_pixel_space: bool = True,
    ):
        """
        Differetiable denormalizer. Output tensor might be in pixel space: [0, 255.0] or unit space: [0, 1.0]
        """
        super().__init__()
        self.to_pixel_space = to_pixel_space
        num_channel = len(mean)

        mean_list = [
            torch.full((input_size, input_size), mean[i], device=device)
            for i in range(num_channel)
        ]  # 3 x [h, w]
        self.mean = torch.unsqueeze(torch.stack(mean_list), 0)  # [1, 3, h, w]

        std_list = [
            torch.full((input_size, input_size), std[i], device=device)
            for i in range(num_channel)
        ]  # 3 x [h, w]
        self.std = torch.unsqueeze(torch.stack(std_list), 0)  # [1, 3, h, w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mul(self.std).add(self.mean)
        if self.to_pixel_space:
            x = x * 255.0
        return x


class PixelModel(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        input_size: int,
        mean: Tuple[float],
        std: Tuple[float],
        device: Union[_device, str, None],
    ):
        """
        Model which take unnormalized pixel space tensor as input.
        """
        super().__init__()
        self.model = model
        self.normalizer = Normalizer(input_size, mean, std, device=device)

    def forward(self, pixel_x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(pixel_x)  # rescale [0, 255] -> [0, 1] and normalize
        return self.model(x)  # IMPORTANT: this return is in [0, 1]


class AttackWrapper(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        mean: Tuple[float],
        std: Tuple[float],
        device: Union[_device, str, None],
    ):
        super().__init__()
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.device = device
        self.normalizer = Normalizer(
            self.input_size, self.mean, self.std, device=self.device
        )
        self.denormalizer = Denormalizer(
            self.input_size, self.mean, self.std, device=self.device
        )

    def forward(
        self, model: torch.nn.Module, x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Return perturbed input in unit space [0,1]
        This function shold be called from all Attacker.
        """
        was_training = model.training
        pixel_model = PixelModel(
            model, self.input_size, self.mean, self.std, self.device
        )
        pixel_model.eval()
        # forward input to  pixel space
        pixel_x = self.denormalizer(x.detach())
        pixel_return = self._forward(pixel_model, pixel_x, *args, **kwargs)
        if was_training:
            pixel_model.train()

        return self.normalizer(pixel_return)
