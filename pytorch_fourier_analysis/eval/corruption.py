import os
import sys
import logging
import hydra
import omegaconf
import torch
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import shared


def calc_mean_error(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str
) -> Tuple[float]:
    """
    Calcurate top1 and top5 error for given model and dataset.
    """
    err1_list, err5_list = list(), list()
    with torch.no_grad():
        for x, t in loader:
            x, t = x.to(device), t.to(device)
            output = model(x)
            err1, err5 = shared.calc_error(output, t, topk=(1, 5))

            err1_list.append(err1.item())
            err5_list.append(err5.item())

    mean_err1 = sum(err1_list) / len(err1_list)
    mean_err5 = sum(err5_list) / len(err5_list)
    return mean_err1, mean_err5


def eval_cifar(
    name: str,
    model: torch.nn.Module,
    root: str,
    transform: torchvision.transforms.transforms.Compose,
    batch_size: int,
    savedir: str,
    device: str,
    corruptions: Iterable[str],
) -> pd.DataFrame:
    """
    Evaluate corruption error by CIFAR-C dataset.

    Args
        name: Name of dataset. (cifar10-c or cifar100-c)
        model: Pretrained model.
        root: Root path to dataset.
        transform: Transform applied to cifar-c dataset.
        batch_size: Size of batch.
        savedir: Directory for saving results.
        device: Device whuich data is sent. (cup or cuda)
        corruptions: Considering corruptions.
    """
    # setup clean cifar dataset.
    dataset_class = shared.get_dataset_class(name.rstrip("-c"), root=root, train=False)
    dataset = dataset_class(transform=transform)

    df = pd.DataFrame(columns=["corruption", "err1", "err5"])

    with tqdm(total=len(corruptions), ncols=80) as pbar:
        for i, corruption_type in enumerate(corruptions):
            # replace clean cifar to currupted one.
            dataset.data = np.load(os.path.join(root, name, corruption_type + ".npy"))
            dataset.targets = torch.LongTensor(
                np.load(os.path.join(root, name, "labels.npy"))
            )

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )

            mean_err1, mean_err5 = calc_mean_error(model, loader, device)

            # append to dataframe
            results = dict(corruption=corruption_type, err1=mean_err1, err5=mean_err5)
            df = df.append(results, ignore_index=True)
            pbar.set_postfix(results)
            pbar.update()

    mean_result = dict(
        corruption="mean", err1=df["err1"].mean(), err5=df["err5"].mean()
    )
    df_wo_noise = df[~df["corruption"].str.endswith("_noise")]
    mean_result_wo_noise = dict(
        corruption="mean_wo_noise",
        err1=df_wo_noise["err1"].mean(),
        err5=df_wo_noise["err5"].mean(),
    )

    df = df.append(mean_result, ignore_index=True)
    df = df.append(mean_result_wo_noise, ignore_index=True)
    return df


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


@hydra.main(config_path="../conf/corruption.yaml")
def main(cfg: omegaconf.DictConfig) -> None:
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
        hydra.utils.get_original_cwd(), "../../data"
    )  # this is needed because hydra automatically change working directory.

    if cfg.dataset.name in ["cifar10-c", "cifar100-c"]:
        df = eval_cifar(
            name=cfg.dataset.name,
            model=model,
            root=dataset_root,
            transform=transform,
            batch_size=cfg.batch_size,
            savedir=cfg.savedir,
            device=device,
            corruptions=cfg.dataset.corruptions,
        )
    elif cfg.dataset.name == "imagenet-c":
        pass

    # save to csv
    df.to_csv("corruption_error_{name}.csv".format(name=cfg.dataset.name))

    # save as plot
    result_dict = dict(zip(df["corruption"], df["err1"]))
    create_barplot(result_dict, "Corruption error", "corruption_error.png")


if __name__ == "__main__":
    main()
