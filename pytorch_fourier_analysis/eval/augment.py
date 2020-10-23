import os
import sys
import pathlib
from typing import Union

import torch
import torchvision
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pytorch_fourier_analysis.mixaugments as mixaugments
import pytorch_fourier_analysis.noiseaugments as noiseaugments


def main(
    method: str, source_img: pathlib.Path, second_img: Union[None, pathlib.Path]
) -> torch.FloatTensor:
    supported_mixaugments = {"cutmix", "cutout", "mixup"}
    model = torchvision.models.resnet50()
    criterion = torch.nn.CrossEntropyLoss()

    transform = torchvision.transforms.Compose(
        [
            # torchvision.transforms.Resize(256),
            # torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ]
    )
    if source_img.exists():
        x_source = Image.open(source_img)
        x_source = transform(x_source)
    else:
        raise ValueError

    if second_img is not None:
        if second_img.exists():
            x_second = Image.open(second_img)
            x_second = transform(x_second)
        else:
            raise ValueError

    input_size = x_source.size(-1)
    if method in supported_mixaugments:
        if method == "cutmix":
            mixaugment = mixaugments.CutMix(alpha=1.0, prob=1.0)
        elif method == "cutout":
            mixaugment = mixaugments.Cutout(prob=1.0, cutout_size=int(input_size / 2))
        elif method == "mixup":
            mixaugment = mixaugments.Mixup(alpha=1.0, prob=1.0)
        else:
            raise NotImplementedError

        x = torch.stack([x_source, x_second], dim=0)
        t = torch.tensor([0, 0], dtype=torch.long)
        _, retdict = mixaugment(model, criterion, x, t)
        return retdict["x"][0]
    else:
        if method == "fourier":
            noiseaugment = noiseaugments.Fourier(prob=1.0, max_index=15, max_eps=16.0)
        elif method == "gaussian":
            noiseaugment = noiseaugments.Gaussian(
                prob=1.0, max_scale=1.0, randomize_scale=True
            )
        elif method == "patch_fourier":
            noiseaugment = noiseaugments.PatchFourier(
                prob=1.0,
                max_index=15,
                max_eps=16.0,
                patch_size=input_size,
                randomize_patch_size=True,
            )
        elif method == "patch_gaussian":
            noiseaugment = noiseaugments.PatchGaussian(
                prob=1.0,
                patch_size=input_size,
                randomize_patch_size=True,
                max_scale=1.0,
                randomize_scale=True,
            )
        else:
            raise NotImplementedError

        return noiseaugment(x_source)


if __name__ == "__main__":
    import argparse
    from argparse import ArgumentDefaultsHelpFormatter

    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", type=str)
    parser.add_argument("--source_img", type=str)
    parser.add_argument("--second_img", type=str)
    parser.add_argument("--log_path", type=str)
    opt = vars(parser.parse_args())

    opt["source_img"] = pathlib.Path(opt["source_img"])
    opt["second_img"] = pathlib.Path(opt["second_img"])

    x = main(opt["method"], opt["source_img"], opt["second_img"])
    torchvision.utils.save_image(x, opt["log_path"])
