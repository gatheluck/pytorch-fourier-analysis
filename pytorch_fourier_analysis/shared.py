import os
import sys
import torch
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pytorch_fourier_analysis import models


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
