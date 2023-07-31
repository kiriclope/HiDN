#!/usr/bin/env python3
import sys

import dual_data.common.constants as gv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dual_data.common.get_data import get_X_y_days, get_X_y_S1_S2
from dual_data.common.options import set_options
from dual_data.common.plot_utils import add_vlines, save_fig
from dual_data.decode.classifiers import get_clf
from dual_data.decode.coefficients import get_coefs
from dual_data.preprocess.helpers import avg_epochs, minmax_X_y
from dual_data.stats.bootstrap import my_boots_ci

# from stats.shuffle import my_shuffle


def get_coef_feat(X_days, y_days, **options):
    model = get_clf(**options)

    if options["features"] == "sample":
        options["task"] = "Dual"
        options["epoch"] = ["ED"]
        options["overlap"] = "sample"
    elif options["features"] == "distractor":
        options["task"] = "Dual"
        options["epoch"] = ["MD"]
        options["overlap"] = "distractor"

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    X_avg = avg_epochs(X_S1_S2, epochs=options["epoch"])
    print("X_avg", X_avg.shape)
    coefs, _ = get_coefs(model, X_avg, y_S1_S2, **options)

    return coefs


def get_overlap(X, coefs):
    overlap = np.zeros((X.shape[-1], X.shape[0]))

    for i_epoch in range(X.shape[-1]):
        overlap[i_epoch] = np.dot(coefs, X[..., i_epoch].T).T

    return -overlap / X.shape[1]


def get_total_overlap(X, y, eps, coefs):
    overlap = np.zeros((X.shape[-1], X.shape[0]))

    idx = np.where(y == 0)[0]

    for i_epoch in range(X.shape[-1]):
        overlap[i_epoch] = np.dot(coefs, X[..., i_epoch].T).T

    A = -np.nanmean(overlap[:, idx], axis=1) / X.shape[1]
    B = -np.nanmean(overlap[:, ~idx], axis=1) / X.shape[1]

    return (A + eps * B) / 2


def run_get_overlap(**kwargs):
    options = set_options(**kwargs)
    task = options["task"]
    trials = options["trials"]

    eps = 1
    options["overlap"] = "distractor"
    if options["features"] == "sample":
        eps = -1
        options["overlap"] = "sample"

    try:
        options["day"] = int(options["day"])
    except:
        pass

    X_days, y_days = get_X_y_days(mouse=options["mouse"], IF_RELOAD=0)

    coefs = get_coef_feat(X_days, y_days, **options)

    options["task"] = task
    options["features"] = "sample"
    options["trials"] = trials

    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    # X = minmax_X_y(X, y)
    print(X.shape, y.shape)

    # overlap = get_overlap(X, coefs, model)

    overlap = get_total_overlap(X, y, eps, coefs)
    _, overlap_ci = my_boots_ci(X, lambda X: get_total_overlap(X, y, -1, coefs))
    # overlap_shuffle = my_shuffle(X_S1_S2, lambda X: get_total_overlap(X, y, coefs))

    plot_overlap(data=overlap, ci=overlap_ci, **options)


def plot_overlap(data, ci=None, **options):
    figname = (
        options["mouse"] + "_" + options["task"] + "_" + options["overlap"] + "_overlap"
    )

    fig = plt.figure(figname)

    if options["day"] == "first":
        pal = sns.color_palette("muted")
    else:
        pal = sns.color_palette("bright")

    paldict = {
        "DPA": pal[3],
        "DualGo": pal[0],
        "DualNoGo": pal[2],
        "Dual": pal[1],
        "all": pal[4],
    }

    # plt.plot(gv.time, np.mean(overlap_A, 1), "--")
    # plt.plot(gv.time, np.mean(overlap_B, 1), "--")
    # plt.plot(gv.time, np.mean(overlap_C, 1), "--")
    # plt.plot(gv.time, np.mean(overlap_D, 1), "--")

    # mean_overlap = (np.mean(overlap_A, 1) + eps * np.mean(overlap_B, 1)) / 2
    # plt.plot(gv.time, mean_overlap, color=paldict[task])

    plt.plot(gv.time, data, color=paldict[options["task"]])
    # plt.plot(time, np.mean(overlap_shuffle, axis=0), '--k')

    plt.plot([0, gv.duration], [0, 0], "--k")
    plt.fill_between(
        gv.time,
        data - ci[:, 0],
        data + ci[:, 1],
        alpha=0.25,
        color=paldict[options["task"]],
    )

    add_vlines()
    plt.xlim([0, 14])

    plt.xlabel("Time (s)")
    plt.ylabel("Overlap")

    save_fig(fig, figname)


if __name__ == "__main__":
    options["features"] = sys.argv[1]
    options["day"] = sys.argv[2]
    options["task"] = sys.argv[3]

    run_get_overlap(**options)
