import pathlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_bandpass_error(df: pd.DataFrame, savepath: pathlib.Path):
    sns.relplot(x="bandwidth", y="err1", hue="method", style="method", dashes=False, markers=True, kind="line", data=df, legend=False)
    plt.savefig(savepath)


if __name__ == "__main__":
    df = pd.read_csv("plots/inputs/bandpass_error_cifar10.csv")
    df = df.drop(["Unnamed: 0", "err5"], axis=1)
    df_low = df[df["filter_mode"] == "low_pass"]
    df_high = df[df["filter_mode"] == "high_pass"]

    # https://note.com/hiro10_yme38/n/nd2fa525942f3#x1qgT
    sns.set()
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    l0 = sns.lineplot(x="bandwidth", y="err1", hue="method", style="method", dashes=False, markers=True, data=df_low, ax=ax[0])
    ax[0].set(xlabel="Bandwidth", ylabel="Error", xlim=(0, 33), ylim=(0, 100))
    ax[0].set_title("Low pass filtered noise")
    handles0, labels0 = ax[0].get_legend_handles_labels()

    l1 = sns.lineplot(x="bandwidth", y="err1", hue="method", style="method", dashes=False, markers=True, data=df_high, ax=ax[1])
    ax[1].set(xlabel="Bandwidth", ylabel="Error", xlim=(0, 33), ylim=(0, 100))
    ax[1].set_title("High pass filtered noise")
    handles1, labels1 = ax[1].get_legend_handles_labels()

    labels0 = [label.capitalize() for label in labels0]

    fig.legend(handles0[1:], labels0[1:], title=None, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    ax[0].legend_.remove()  # remove default legend
    ax[1].legend_.remove()  # remove default legend

    savepath = pathlib.Path("plots/outputs/bandpass_error_cifar10.png")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(savepath)

    # for name, df in {"lowpass": df_low, "highpass": df_high}.items():
    #     # https://note.com/hiro10_yme38/n/nd2fa525942f3#x1qgT
    #     sns.set()
    #     sns.set_style("whitegrid")

    #     fig, ax = plt.subplots(figsize=(7, 5))

    #     ax = sns.lineplot(x="bandwidth", y="err1", hue="method", style="method", dashes=False, markers=True, data=df, legend=False)
    #     ax.set(xlabel="bandwidth", ylabel="error", xlim=(0, 33), ylim=(0, 100))

    #     savepath = pathlib.Path("plots/outputs/bandpass_error_cifar10_{filter_mode}.png".format(filter_mode=name))
    #     plt.savefig(savepath)

    print(df_low)
    print(df_high)