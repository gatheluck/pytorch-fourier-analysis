import os
import glob
from typing_extensions import Final

import numpy as np
import pandas as pd


def main(target_dir: str, tex_mode: bool, **kwargs):
    target_path = os.path.join(target_dir, "**", "*.csv")
    csvpaths: Final = sorted(glob.glob(target_path, recursive=True))
    for csvpath in csvpaths:
        print(csvpath)
        df = pd.read_csv(csvpath, index_col="Unnamed: 0")
        clean = df.at[17, "err1"]
        df = df.iloc[0:-3, :]  # ignore clean, mean, mean w/o noise
        df_wo_gauss = df[df["corruption"] != "gaussian_noise"]
        print(df["corruption"].to_list())

        plus_minus = "$\pm$" if tex_mode else " + "

        print(" ")
        print(
            "claen: {clean:0.1f}, mCE: {mean:0.1f}{plus_minus}{std:0.1f}, mCE(-gauss): {mean_wo_gauss:0.1f}{plus_minus_}{std_wo_gauss:0.1f}".format(
                clean=clean,
                mean=df["err1"].mean(),
                plus_minus=plus_minus,
                std=np.sqrt(df["err1"].var()),
                mean_wo_gauss=df_wo_gauss["err1"].mean(),
                plus_minus_=plus_minus,
                std_wo_gauss=np.sqrt(df_wo_gauss["err1"].var()),
            )
        )
        print(" ")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-t", "--target_dir", type=str, help="target directory")
    parser.add_argument("--tex_mode", action="store_true", default=False)
    opt = parser.parse_args()

    main(**vars(opt))
