import os
import sys
import copy
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


def eval_cifar(
    root: str,
    transform: torchvision.transforms.transforms.Compose,
    batch_size: int,
    norm: str,
    savedir: str,
    device: str,
    corruptions: Iterable[str],
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
    name = "cifar10-c"

    # setup clean cifar dataset.
    dataset_class = shared.get_dataset_class(name.rstrip("-c"), root=root, train=False)
    dataset = dataset_class(transform=transform)

    dataset_long = copy.deepcopy(dataset)
    dataset_long.data = np.tile(dataset.data, (5, 1, 1, 1))
    dataset_long.targets = dataset.targets * 5

    df = pd.DataFrame(columns=["corruption", "norm"])

    # clean dataset
    loader_clean = torch.utils.data.DataLoader(
        dataset_long,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    with tqdm(total=len(corruptions), ncols=80) as pbar:
        # corruption dataset
        for i, corruption_type in enumerate(corruptions):
            # replace clean cifar to currupted one
            dataset.data = np.load(os.path.join(root, name, corruption_type + ".npy"))
            dataset.targets = torch.LongTensor(
                np.load(os.path.join(root, name, "labels.npy"))
            )

            loader_corruption = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )

            mean_norm_list = list()
            for (x_clean, _), (x_curruption, _) in zip(loader_clean, loader_corruption):
                if norm == 'l2':
                    mean_norm = (x_curruption - x_clean).norm(p=2).item()
                elif norm == 'linf':
                    mean_norm = (x_curruption - x_clean).norm(p=float('inf')).item()
                else:
                    raise NotImplementedError

                mean_norm_list.append(mean_norm)

                # x_clean_for_save = x_clean[0:8]
                # x_curruption_for_save = x_curruption[0:8]
                # x_for_save = torch.cat([x_clean_for_save, x_curruption_for_save, x_curruption_for_save - x_clean_for_save], dim=-2)
                # torchvision.utils.save_image(x_for_save, "test_norm.png")
                # raise NotADirectoryError

            # append to dataframe
            result = dict(corruption=corruption_type, norm=sum(mean_norm_list) / float(len(mean_norm_list)))
            df = df.append(result, ignore_index=True)
            pbar.set_postfix(result)
            pbar.update()

        # # calculate mean result
        # mean_result = dict(
        #     corruption="mean", err1=df["err1"].mean(), err5=df["err5"].mean()
        # )
        # df_wo_noise = df[~df["corruption"].str.endswith("_noise")]
        # mean_result_wo_noise = dict(
        #     corruption="mean_wo_noise",
        #     err1=df_wo_noise["err1"].mean(),
        #     err5=df_wo_noise["err5"].mean(),
        # )

        # df = df.append(mean_result, ignore_index=True)
        # df = df.append(mean_result_wo_noise, ignore_index=True)
        # df = df.append(
        #     clean_result, ignore_index=True
        # )  # append here to prevent having effect to mean result
    return df


def create_barplot(errs: Dict[str, float], title: str, savepath: str):
    y = list(errs.values())
    x = np.arange(len(y))
    xticks = list(errs.keys())

    plt.bar(x, y)
    for i, j in zip(x, y):
        plt.text(i, j, f"{j:.1f}", ha="center", va="bottom", fontsize=7)

    plt.title(title)
    plt.ylabel("norm")

    plt.ylim(0, max(errs.values()))

    plt.xticks(x, xticks, rotation=90)
    # plt.yticks(np.linspace(0, 100, 11))

    plt.subplots_adjust(bottom=0.3)
    plt.grid(axis="y")
    plt.savefig(savepath)
    plt.close()


@hydra.main(config_path="../conf/corruption_norm.yaml")
def main(cfg: omegaconf.DictConfig) -> None:
    # setup device
    device = "cuda" if cfg.gpus > 0 else "cpu"

    # setup dataset
    transform = shared.get_transform(
        cfg.dataset.input_size, cfg.dataset.mean, cfg.dataset.std, train=False
    )
    dataset_root = os.path.join(
        hydra.utils.get_original_cwd(), "data"
    )  # this is needed because hydra automatically change working directory.

    df = eval_cifar(
        root=dataset_root,
        transform=transform,
        batch_size=cfg.batch_size,
        norm=cfg.norm,
        savedir=cfg.savedir,
        device=device,
        corruptions=cfg.dataset.corruptions,
    )

    # save to csv
    df.to_csv("corruption_norm_{name}_{norm}.csv".format(name=cfg.dataset.name, norm=cfg.norm))

    # save as plot
    result_dict = dict(zip(df["corruption"], df["norm"]))
    create_barplot(result_dict, "mean corruption norm", "corruption_norm.png")


if __name__ == "__main__":
    main()