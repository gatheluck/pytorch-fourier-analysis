import os
import sys
import random
from typing import Dict, Iterable, Tuple

import hydra
import omegaconf
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import shared
from pytorch_fourier_analysis import fourier
from pytorch_fourier_analysis import attacks


def calc_mean_error(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    normalizer,
    denormalizer,
    device: str,
    bandwidth: int,
    filter_mode: str,
    eps: float,
) -> Tuple[float]:
    """
    Calcurate top1 and top5 error for given model and dataset.
    """
    err1_list, err5_list = list(), list()
    with torch.no_grad():
        for i, (x, t) in enumerate(loader):
            x, t = x.to(device), t.to(device)

            scale_r = random.uniform(0, 1)
            scale_g = random.uniform(0, 1)
            scale_b = random.uniform(0, 1)
            gaussian_r = torch.normal(mean=0.0, std=scale_r, size=[x.size(0), x.size(-2), x.size(-1)]).to(device)
            gaussian_g = torch.normal(mean=0.0, std=scale_g, size=[x.size(0), x.size(-2), x.size(-1)]).to(device)
            gaussian_b = torch.normal(mean=0.0, std=scale_b, size=[x.size(0), x.size(-2), x.size(-1)]).to(device)
            gaussian = torch.stack([gaussian_r, gaussian_g, gaussian_b], dim=1)

            if 0.0 < bandwidth < x.size(-1):
                # scale = random.uniform(0, 1)
                # gaussian = torch.normal(mean=0.0, std=scale, size=list(x.size())).to(device)
                bandpassed_gaussian, w = fourier.bandpass_filter(gaussian, bandwidth, filter_mode, eps)
                x = torch.clamp(denormalizer(x) + bandpassed_gaussian, 0.0, 1.0)
                x = normalizer(x)
            elif bandwidth == x.size(-1):
                # scale = random.uniform(0, 1)
                # gaussian = torch.normal(mean=0.0, std=scale, size=list(x.size())).to(device)

                norms_r = gaussian[:, 0, :, :].view(gaussian.size(0), -1).norm(dim=-1)  # (B)
                norms_g = gaussian[:, 1, :, :].view(gaussian.size(0), -1).norm(dim=-1)  # (B)
                norms_b = gaussian[:, 2, :, :].view(gaussian.size(0), -1).norm(dim=-1)  # (B)

                gaussian[:, 0, :, :] /= norms_r[:, None, None]
                gaussian[:, 1, :, :] /= norms_g[:, None, None]
                gaussian[:, 2, :, :] /= norms_b[:, None, None]
                gaussian *= eps

                x = torch.clamp(denormalizer(x) + gaussian, 0.0, 1.0)
                x = normalizer(x)

            output = model(x)
            err1, err5 = shared.calc_error(output, t, topk=(1, 5))

            err1_list.append(err1.item())
            err5_list.append(err5.item())

            if (bandwidth == 0) or (bandwidth == x.size(-1)):
                x_sample = None
            elif i == 0:
                x_sample = torch.cat([denormalizer(x), gaussian, bandpassed_gaussian, w], dim=-2)[0:16]

    mean_err1 = sum(err1_list) / len(err1_list)
    mean_err5 = sum(err5_list) / len(err5_list)
    return mean_err1, mean_err5, x_sample


def eval_gaussian_noise_error(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    normalizer: torch.nn.Module,
    denormalizer: torch.nn.Module,
    savedir: str,
    device: str,
    low_bandwidths: Iterable[int],
    high_bandwidths: Iterable[int],
    eps: float,
) -> pd.DataFrame:
    """
    Evaluate corruption error by CIFAR-C dataset.

    Args
        name: Name of dataset (cifar10-c or cifar100-c)
        model: Pretrained model
        root: Root path to dataset
        transform: Transform applied to cifar-c dataset
        batch_size: Size of batch
        savedir: Directory for saving results
        device: Device whuich data is sent (cpu or cuda)
        corruptions: Considering corruptions
    """
    # create dataframe
    df = pd.DataFrame(columns=["filter_mode", "bandwidth", "err1", "err5"])

    # low pass filter
    with tqdm(total=len(low_bandwidths), ncols=80) as pbar:
        for bandwidth in low_bandwidths:
            mean_err1, mean_err5, x_sample = calc_mean_error(
                model, loader, normalizer, denormalizer, device, bandwidth, "low_pass", eps
            )
            result = dict(
                filter_mode="low_pass", bandwidth=bandwidth, err1=mean_err1, err5=mean_err5
            )
            df = df.append(result, ignore_index=True)
            pbar.set_postfix(result)
            pbar.update()
            if x_sample is not None:
                torchvision.utils.save_image(x_sample, "lowpass_bandwidth-{bandwidth}.png".format(bandwidth=bandwidth))

    # high pass filter
    with tqdm(total=len(high_bandwidths), ncols=80) as pbar:
        for bandwidth in high_bandwidths:
            mean_err1, mean_err5, x_sample = calc_mean_error(
                model, loader, normalizer, denormalizer, device, bandwidth, "high_pass", eps
            )
            result = dict(
                filter_mode="high_pass", bandwidth=bandwidth, err1=mean_err1, err5=mean_err5
            )
            df = df.append(result, ignore_index=True)
            pbar.set_postfix(result)
            pbar.update()
            if x_sample is not None:
                torchvision.utils.save_image(x_sample, "highpass_bandwidth-{bandwidth}.png".format(bandwidth=bandwidth))

    return df


@hydra.main(config_path="../conf/bandpass.yaml")
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

    # setup dataset
    transform = shared.get_transform(
        cfg.dataset.input_size, cfg.dataset.mean, cfg.dataset.std, train=False
    )
    dataset_root = os.path.join(
        hydra.utils.get_original_cwd(), "data"
    )  # this is needed because hydra automatically change working directory.

    dataset_class = shared.get_dataset_class(
        name=cfg.dataset.name, root=dataset_root, train=False
    )
    dataset = dataset_class(transform=transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    normalizer = attacks.Normalizer(input_size=cfg.dataset.input_size,
                                    mean=cfg.dataset.mean,
                                    std=cfg.dataset.std,
                                    device=device,
                                    from_pixel_space=False)
    denormalizer = attacks.Denormalizer(input_size=cfg.dataset.input_size,
                                        mean=cfg.dataset.mean,
                                        std=cfg.dataset.std,
                                        device=device,
                                        to_pixel_space=False)

    df = eval_gaussian_noise_error(
        model,
        loader,
        normalizer,
        denormalizer,
        cfg.savedir,
        device,
        cfg.low_bandwidths,
        cfg.high_bandwidths,
        cfg.eps,
    )

    # save to csv
    df.to_csv("bandpass_error_{name}.csv".format(name=cfg.dataset.name))

    df_highpass = df[df["filter_mode"] == "high_pass"]
    df_lowpass = df[df["filter_mode"] == "low_pass"]

    # save as plot
    result_dict_highpass = dict(zip(df_highpass["bandwidth"], df_highpass["err1"]))
    result_dict_lowpass = dict(zip(df_lowpass["bandwidth"], df_lowpass["err1"]))

    create_barplot(result_dict_highpass, "", "error_highpass.png")
    create_barplot(result_dict_lowpass, "", "error_lowpass.png")


def create_barplot(errs: Dict[str, float], title: str, savepath: str):
    y = list(errs.values())
    x = np.arange(len(y))
    xticks = list(errs.keys())

    plt.bar(x, y)
    for i, j in zip(x, y):
        plt.text(i, j, f"{j:.1f}", ha="center", va="bottom", fontsize=7)

    plt.title(title)
    plt.ylabel("Error (%)")

    plt.ylim(0, 100)

    plt.xticks(x, xticks, rotation=90)
    plt.yticks(np.linspace(0, 100, 11))

    plt.subplots_adjust(bottom=0.3)
    plt.grid(axis="y")
    plt.savefig(savepath)
    plt.close()


if __name__ == "__main__":
    main()
