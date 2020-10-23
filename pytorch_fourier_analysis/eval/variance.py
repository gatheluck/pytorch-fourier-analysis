import os
import glob
from typing_extensions import Final

import numpy as np
import pandas as pd


def main(target_dir: str, drop_gauss: bool, **kwargs):
    target_path = os.path.join(target_dir, "**", "*.csv")
    csvpaths: Final = sorted(glob.glob(target_path, recursive=True))
    for csvpath in csvpaths:
        print(csvpath)
        df = pd.read_csv(csvpath, index_col="Unnamed: 0")
        df = df.iloc[0:-3, :]
        df = df[df["corruption"] != "gaussian_noise"] if drop_gauss else df
        print(df["corruption"].to_list())
        print(
            "mean:{mean:0.2f}, std:{std:0.2f}".format(
                mean=df["err1"].mean(), std=np.sqrt(df["err1"].var())
            )
        )
        print(" ")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-t", "--target_dir", type=str, help="target directory")
    parser.add_argument("--drop_gauss", action="store_true", default=False)
    opt = parser.parse_args()

    main(**vars(opt))
