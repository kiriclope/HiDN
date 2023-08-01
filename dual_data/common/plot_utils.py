import copy
import os
import pickle as pkl
import string

import dual_data.common.constants as gv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

golden_ratio = (5**0.5 - 1) / 2
width = 7
matplotlib.rcParams["figure.figsize"] = [width, width * golden_ratio]


def add_vlines(ax=None):
    time_periods = [gv.t_STIM, gv.t_DIST, gv.t_TEST, gv.t_CUE]
    colors = ["b", "b", "b", "g"]
    if ax is None:
        for period, color in zip(time_periods, colors):
            plt.axvspan(period[0], period[1], alpha=0.1, color=color)
    else:
        for period, color in zip(time_periods, colors):
            ax.axvspan(period[0], period[1], alpha=0.1, color=color)


def save_fig(fig, figname, path=gv.figdir, format="svg", dpi="figure"):
    if not os.path.isdir(path):
        os.makedirs(path)

    pkl.dump(fig, open(path + "/" + figname + ".pkl", "wb"))
    plt.savefig(path + "/" + figname + "." + format, dpi=dpi, format=format)


def pkl_save(obj, name, path="."):
    pkl.dump(obj, open(path + "/" + name + ".pkl", "wb"))


def pkl_load(name, path="."):
    return pkl.load(open(path + "/" + name, "rb"))


def copy_fig(fig, ax, vline=0):
    ax0 = fig.axes[0]

    # labels
    x_label = ax0.xaxis.get_label().get_text()
    y_label = ax0.yaxis.get_label().get_text()
    # print(x_label, y_label)

    # lines
    lines = ax0.get_lines()
    for line in lines:
        xydata = line.get_xydata()
        color = line.get_c()
        ls = line.get_linestyle()
        marker = line.get_marker()
        ax.plot(xydata[:, 0], xydata[:, 1], marker=marker, color=color, ls=ls)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    collects = ax0.collections
    # print(collects)
    for collect in collects:
        collect_cpy = copy.copy(collect)
        collect_cpy.axes = None
        collect_cpy.figure = None
        collect_cpy.set_transform(ax.transData)
        ax.add_collection(collect_cpy)

    # ticks
    x_ticks = ax0.get_xaxis().properties()["ticklocs"]
    y_ticks = ax0.get_yaxis().properties()["ticklocs"]

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # limits
    x_lim = ax0.get_xaxis().properties()["view_interval"]
    y_lim = ax0.get_yaxis().properties()["view_interval"]

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if vline:
        add_vlines(ax)
    # ax.autoscale()


def concat_fig(
    figname,
    figlist,
    dim=[1, 2],
    size=[2.427, 1.5],
    VLINE=1,
    LABEL=1,
    LABEL_POS=[-0.2, 1.2],
):
    fig, ax = plt.subplots(
        dim[0],
        dim[1],
        figsize=(size[0] * dim[1], size[1] * dim[0]),
        num=figname,
    )

    fig_iter = iter(figlist)
    count = 0

    labels = list(string.ascii_uppercase[: len(figlist)])

    if np.array(ax.shape).shape != (1,):
        for col in range(ax.shape[0]):
            for row in range(ax.shape[1]):
                if count < len(figlist):
                    copy_fig(next(fig_iter), ax[col][row], VLINE)

                    if LABEL:
                        ax[col][row].text(
                            LABEL_POS[0],
                            LABEL_POS[1],
                            labels[count],
                            transform=ax[col][row].transAxes,
                            va="top",
                            ha="right",
                            weight="bold",
                        )

                    count += 1
    else:
        for row in range(ax.shape[0]):
            if count < len(figlist):
                copy_fig(next(fig_iter), ax[row], VLINE)
                if LABEL:
                    ax[count].text(
                        LABEL_POS[0],
                        LABEL_POS[1],
                        labels[count],
                        transform=ax[count].transAxes,
                        va="top",
                        ha="right",
                        weight="bold",
                    )

                count += 1

    plt.tight_layout()
    plt.show()
    # return fig


def plot_summary(pickle_files, H, W, figsize):
    fig, axs = plt.subplots(H, W, figsize=figsize)

    for i, ax in enumerate(axs.flatten()):
        with open(pickle_files[i], "rb") as f:
            fig = pkl.load(f)
            ax.axis("off")  # do not show axis value
            # Display the figure
            ax.imshow(fig)

    plt.tight_layout(True)
    plt.show()


if __name__ == "__main__":
    fig = plt.figure()
    x = np.linspace(0, 2 * np.pi)

    y = np.sin(x)
    plt.plot(x, y)
    plt.ylabel("sin(x)")
    plt.xlabel("x")

    pkl_save(fig, "sin")
    plt.close(fig)

    fig = plt.figure()
    y = np.cos(x)
    plt.plot(x, y)
    plt.fill_between(x, y - 1, y + 1, alpha=0.25)
    plt.ylabel("cos(x)")
    plt.xlabel("x")

    pkl_save(fig, "cos")
    plt.close(fig)

    cos = pkl_load("cos.pkl")
    sin = pkl_load("sin.pkl")

    # plt.close('all')

    fig_list = [cos, sin, cos, sin]
    concat_fig("summary", fig_list, dim=[2, 2])
