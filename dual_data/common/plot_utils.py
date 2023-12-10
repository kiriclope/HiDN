import os
import pickle as pkl
import string

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

import dual_data.common.constants as gv

sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

golden_ratio = (5**0.5 - 1) / 2
width = 7
matplotlib.rcParams["figure.figsize"] = [width, width * golden_ratio]

from string import ascii_uppercase

import svgwrite
from IPython.display import SVG, display
from svgutils.transform import SVGFigure, fromfile


def create_svg_grid(image_files, grid_dims):
    # labeled_files = []
    # for idx, image_file in enumerate(image_files):
    #     fig = fromfile(image_file)
    #     label_text = ascii_uppercase[idx]
    #     text_fig = SVGFigure().text(
    #         label_text, (10, 20)
    #     )  # (10, 20): adjust this according to your svg size
    #     fig.getroot().append(text_fig)  # append text
    #     labeled_svg = f"labeled_{image_file}"
    #     fig.save(labeled_svg)
    #     labeled_files.append(labeled_svg)

    grid_image = svgwrite.Drawing("grid_image.svg")
    rows, cols = grid_dims

    for idx, labeled_svg in enumerate(image_files):
        x = idx % cols * 200  # 200: assuming each image width
        y = idx // cols * 200  # 200: assuming each image height
        image = svgwrite.image.Image(labeled_svg, insert=(x, y))
        grid_image.add(image)

    grid_image.save("grid_image.svg")
    display(SVG(filename="grid_image.svg"))


def add_vlines(ax=None, mouse=""):
    t_BL = [0, 2]
    t_STIM = [2 , 3]
    t_ED = [3 , 4.5]
    t_DIST = [4.5 , 5.5]
    t_MD = [5.5 , 6.5]
    t_CUE = [6.5 , 7]
    t_RWD = [7 , 7.5]
    t_LD = [7.5 , 9]
    t_TEST = [9 , 10]
    t_RWD2 = [11 , 12]
    
    if "P" in mouse:
        t_BL = [0 , 2]
        t_STIM = [2 , 3]
        t_ED = [3 , 4]
        t_DIST = [4 , 5]
        t_MD = [5 , 6]
        t_CUE = [6 , 7]
        t_RWD = [7 , 8]
        t_LD = [8 , 9]
        t_TEST = [9 , 10]
        t_RWD2 = [10 , 11]

    time_periods = [t_STIM, t_DIST, t_TEST, t_CUE]
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

    # pkl.dump(fig, open(path + "/" + figname + ".pkl", "wb"))
    plt.savefig(path + "/" + figname + "." + format, dpi=dpi, format=format)
    
    # format = "png"
    # plt.savefig(path + "/" + figname + "." + format, dpi=dpi, format=format)


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

    # im = ax0.images[0]
    # cbar = im.colorbar
    # cax = ax.imshow(
    #     im.get_array(),
    #     cmap=im.get_cmap(),
    #     vmin=im.get_clim()[0],
    #     vmax=im.get_clim()[1],
    #     extent=im.get_extent(),
    #     origin="lower",
    # )

    # xlim = ax0.get_xlim()
    # ylim = ax0.get_ylim()
    # xticks = ax0.get_xticks()
    # yticks = ax0.get_yticks()

    # ax.figure.colorbar(cax, ax=ax, label=cbar.ax.get_ylabel(), ticks=cbar.get_ticks())

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
    xticks = ax0.get_xaxis().properties()["ticklocs"]
    yticks = ax0.get_yaxis().properties()["ticklocs"]

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # limits
    xlim = ax0.get_xaxis().properties()["view_interval"]
    ylim = ax0.get_yaxis().properties()["view_interval"]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Set the title to the new axes
    title = ax0.get_title()
    ax.set_title(title)

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


def copy_to_png(figname, ax):
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    # Get the size in inches
    axs_width, axs_height = extent.width, extent.height

    # Convert to pixels
    axs_width *= fig.dpi
    axs_height *= fig.dpi

    img = mpimg.imread(figname)
    ax.imshow(img)
    ax.axis("off")


def plot_grid(
    figname,
    figlist,
    dim=[1, 2],
    size=[2.427, 1.5],
    LABEL=1,
    LABEL_POS=[-0.2, 1.2],
):
    fig, axs = plt.subplots(
        dim[0],
        dim[1],
        figsize=(size[0] * dim[1], size[1] * dim[0]),
        num=figname,
    )

    images = [Image.open(img_path) for img_path in figlist]

    # extent = axs[0][0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # axs_width, axs_height = extent.width, extent.height
    # axs_width *= fig.dpi  # Convert to pixels
    # axs_height *= fig.dpi  # Convert to pixels
    # axs_size = (int(axs_width), int(axs_height))  # Round off to the nearest pixel

    # # Resize images to fit the subplot axes
    # resized_images = [img.resize(axs_size) for img in images]
    # np_images = [np.array(img) for img in resized_images]

    labels = list(string.ascii_uppercase[: len(figlist)])
    count = 0
    for i, ax in enumerate(axs.flat):
        ax.imshow(
            images[i],
            extent=ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()),
        )
        # ax.imshow(np_images[i])
        ax.axis("off")
        if LABEL:
            ax.text(
                LABEL_POS[0],
                LABEL_POS[1],
                labels[count],
                transform=ax.transAxes,
                va="top",
                ha="right",
                weight="bold",
            )
        count += 1
    plt.tight_layout()
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
