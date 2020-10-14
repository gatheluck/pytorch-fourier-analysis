import os
import sys
import re
import logging
import functools
from collections import OrderedDict
from typing import Any, List, Tuple, Union

import omegaconf
import numpy as np
import torch
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pytorch_fourier_analysis import models
from pytorch_fourier_analysis import mixaugments
from pytorch_fourier_analysis import noiseaugments
from pytorch_fourier_analysis.mixaugments.base import MixAugmentationBase
from pytorch_fourier_analysis.noiseaugments.base import NoiseAugmentationBase


def get_model(name: str, num_classes: int, inplace: bool = True) -> torch.nn.Module:
    # select model
    if name == "resnet50":
        model = torchvision.models.resnet50(pretrained=False, num_classes=num_classes)
    elif name == "wideresnet40":
        model = models.wideresnet40(num_classes=num_classes, widening_factor=2)
    else:
        raise NotImplementedError

    # relpace ReLU.inplace
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU):
            m.inplace = inplace

    return model


def save_model(model: torch.nn.Module, path: str) -> None:
    torch.save(
        model.module.state_dict()
        if isinstance(model, torch.nn.DataParallel)
        else model.state_dict(),
        path,
    )


def load_model(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError('path "{path}" does not exist.'.format(path=path))
    logging.info("loading model weight from {path}".format(path=path))

    # load weight from .pth file.
    if path.endswith(".pth"):
        weight = torch.load(path)
        statedict = OrderedDict(
            [(re.sub("^module.", "", k), v) for k, v in weight.items()]
        )
        model.load_state_dict(statedict)
    # load weight from checkpoint.
    elif path.endswith(".ckpt"):
        checkpoint = torch.load(path)
        if "state_dict" in checkpoint.keys():
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            raise ValueError("this checkponint do not inculdes state_dict")
    else:
        raise ValueError("path is not supported type of extension.")


def get_dataset_class(name: str, root: str, train: bool):
    _built_in_datasets = set(["cifar10", "cifar100"])
    _root = os.path.join(root, name)

    if name in _built_in_datasets:
        if name == "cifar10":
            dataset_class = functools.partial(
                torchvision.datasets.CIFAR10, root=_root, train=train, download=True
            )
        elif name == "cifar100":
            dataset_class = functools.partial(
                torchvision.datasets.CIFAR100, root=_root, train=train, download=True
            )
        else:
            raise NotImplementedError

    else:
        _root = os.path.join(_root, "train" if train else "val")
        dataset_class = functools.partial(torchvision.datasets.ImageFolder, root=_root)

    return dataset_class


def get_transform(
    input_size: int,
    mean: List[float],
    std: List[float],
    train: bool,
    normalize: bool = True,
    optional_transform: List[Any] = [],
) -> torchvision.transforms.transforms.Compose:
    transform = list()

    # apply standard data augmentation
    if input_size == 32:
        if train:
            transform.extend(
                [
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomCrop(32, 4),
                ]
            )
        else:
            pass
    elif input_size == 224:
        if train:
            transform.extend(
                [
                    torchvision.transforms.RandomResizedCrop(224),
                    torchvision.transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            transform.extend(
                [
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(224),
                ]
            )
    else:
        raise NotImplementedError

        transform.extend(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    # to tensor
    transform.extend([torchvision.transforms.ToTensor()])

    # optional (Fourier Noise, Patch Shuffle, etc.)
    if optional_transform:
        transform.extend(optional_transform)

    # normalize
    if normalize:
        transform.extend([torchvision.transforms.Normalize(mean=mean, std=std)])

    return torchvision.transforms.Compose(transform)


def get_optimizer_class(cfg: omegaconf.DictConfig) -> torch.optim.Optimizer:
    _cfg = omegaconf.OmegaConf.to_container(cfg)
    name = _cfg.pop("name")  # without pop raise KeyError.

    if name == "sgd":
        optimizer_class = torch.optim.SGD
    else:
        raise NotImplementedError

    return functools.partial(optimizer_class, **_cfg)


def get_scheduler_class(
    cfg: omegaconf.DictConfig, cosin_annealing_func=None
) -> torch.optim.lr_scheduler._LRScheduler:
    _cfg = omegaconf.OmegaConf.to_container(cfg)
    name = _cfg.pop("name")  # without pop raise KeyError.

    if name == "multisteplr":
        scheduler_class = torch.optim.lr_scheduler.MultiStepLR
    elif name == "cosinlr":
        scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
    elif name == "cosin_annealing":
        scheduler_class = torch.optim.lr_scheduler.LambdaLR
        scheduler_class = functools.partial(
            scheduler_class, lr_lambda=cosin_annealing_func
        )
    else:
        raise NotImplementedError

    return functools.partial(scheduler_class, **_cfg)


def get_mixaugment(cfg: omegaconf.DictConfig) -> Union[None, MixAugmentationBase]:
    _cfg = omegaconf.OmegaConf.to_container(cfg)
    name = _cfg.pop("name")  # without pop raise KeyError.

    if name is None:
        mixaugment = None
    elif name == "cutmix":
        mixaugment = mixaugments.CutMix(**_cfg)
    elif name == "cutout":
        mixaugment = mixaugments.Cutout(**_cfg)
    elif name == "mixup":
        mixaugment = mixaugments.Mixup(**_cfg)
    else:
        raise NotImplementedError

    return mixaugment


def get_noiseaugment(cfg: omegaconf.DictConfig) -> Union[None, NoiseAugmentationBase]:
    _cfg = omegaconf.OmegaConf.to_container(cfg)
    name = _cfg.pop("name")  # without pop raise KeyError.

    if name is None:
        noiseaugment = None
    elif name == "gaussian":
        noiseaugment = noiseaugments.Gaussian(**_cfg)
    elif name == "patch_gaussian":
        noiseaugment = noiseaugments.PatchGaussian(**_cfg)
    else:
        raise NotImplementedError

    return noiseaugment


def calc_error(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)
) -> List[torch.Tensor]:
    """
    Calculate top-k errors.

    Args
        output: Output tensor from model.
        target: Training target tensor.
        topk: Tuple of int which you want to now error.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(
            maxk, dim=1
        )  # return the k larget elements. top-k index: size (b, k).
        pred = pred.t()  # (k, b)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        errors = list()
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            wrong_k = batch_size - correct_k
            errors.append(wrong_k.mul_(100.0 / batch_size))

        return errors


def cosin_annealing(step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + np.cos(step / total_steps * np.pi))
