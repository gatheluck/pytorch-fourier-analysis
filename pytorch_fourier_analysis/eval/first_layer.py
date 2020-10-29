import os
import sys
import hydra
import omegaconf
import logging
import pathlib
from typing import List

import torch
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pytorch_fourier_analysis.shared as shared


def extract_target_modules(
    model: torch.nn.Module,
    target_module_name: str = "torch.nn.Conv2d",
    verbose: bool = True,
) -> List[torch.nn.Module]:
    """
    return list of specified modules which is included in the given model.
    """
    model.eval()

    target_modules = [
        module
        for module in model.modules()
        if isinstance(module, eval(target_module_name))
    ]

    # log info
    if verbose:
        logging.info(
            "extract_target_module: found {num} [{name}] modules.".format(
                num=len(target_modules), name=target_module_name
            )
        )
    return target_modules


def save_first_layer_weight(
    model: torch.nn.Module,
    log_path: pathlib.Path,
    bias: float = 0.5,
    verbose: bool = True,
) -> None:
    """
    save weight of first conv as images.
    """
    model.eval()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    conv2d_modules = extract_target_modules(model, target_module_name="torch.nn.Conv2d")
    first_conv2d_weight = conv2d_modules[0].weight + bias

    torchvision.utils.save_image(first_conv2d_weight, log_path, padding=1)

    # log info
    if verbose:
        logging.info(
            "save_first_layer_weight: images are saved under [{log_dir}]".format(
                log_dir=log_path.parent
            )
        )


@hydra.main(config_path="../conf/first_layer.yaml")
def main(cfg: omegaconf.DictConfig) -> None:
    """
    Main entry point function to evaluate corruption robustness.
    """
    # setup device
    device = "cuda" if cfg.gpus > 0 else "cpu"

    # setup model
    model = shared.get_model(name=cfg.arch, num_classes=cfg.dataset.num_classes)
    shared.load_model(model, cfg.weight)
    model = model.to(device)
    model.eval()

    save_first_layer_weight(
        model, pathlib.Path(cfg.logdir) / pathlib.Path("first_layer_weight.png")
    )


if __name__ == "__main__":
    main()
