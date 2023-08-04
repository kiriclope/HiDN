#!/usr/bin/env python3
from statistics.bootstrap import my_boots_ci

import common.constants as gv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from common.options import set_options
from common.plot_utils import pkl_save, save_fig
from data.get_data import get_X_y_days, get_X_y_S1_S2
from decode.classifiers import get_clf
from decode.coefficients import get_coefs
from imblearn.over_sampling import SVMSMOTE
from preprocess.augmentation import spawner
from preprocess.helpers import avg_epochs, preprocess_X

# from statistics.shuffle import my_shuffle


def add_vlines(ax=None):
    time_periods = [gv.t_STIM, gv.t_DIST, gv.t_TEST, gv.t_CUE]
    colors = ["b", "b", "b", "g"]
    if ax is None:
        for period, color in zip(time_periods, colors):
            plt.axvspan(period[0], period[1], alpha=0.1, color=color)
    else:
        for period, color in zip(time_periods, colors):
            ax.axvspan(period[0], period[1], alpha=0.1, color=color)


def get_overlap(X, y, coefs, RETURN_SAMPLE=None):

    if coefs.ndim > 1:
        overlap = np.zeros((X.shape[-1], X.shape[0], 4))
    else:
        overlap = np.zeros((X.shape[-1], X.shape[0]))

    print("overlap", overlap.shape)

    for i_epoch in range(X.shape[-1]):
        overlap[i_epoch] = np.dot(coefs, X[..., i_epoch].T).T

    # return -overlap / X.shape[1]

    # averaging over trials
    idx = np.where(y == 0)[0]
    overlap_A = np.nanmean(overlap[:, idx], axis=1) / X.shape[1]
    overlap_B = np.nanmean(overlap[:, ~idx], axis=1) / X.shape[1]

    if RETURN_SAMPLE == "A":
        return -overlap_A
    elif RETURN_SAMPLE == "B":
        return -overlap_B
    else:
        return -overlap_A, -overlap_B


def get_mean_overlap(X, y, coefs, sign=1):

    if coefs.ndim > 1:
        overlap = np.zeros((X.shape[-1], X.shape[0], 4))
    else:
        overlap = np.zeros((X.shape[-1], X.shape[0]))

    idx = np.where(y == 0)[0]

    for i_epoch in range(X.shape[-1]):
        overlap[i_epoch] = np.dot(coefs, X[..., i_epoch].T).T

    A = np.nanmean(overlap[:, idx], axis=1) / X.shape[1] / 2
    B = np.nanmean(overlap[:, ~idx], axis=1) / X.shape[1] / 2

    return A + sign * B


def plot_overlap(day="first", overlap="sample", features="sample", IF_RELOAD=False):
    options = set_options()

    options["day"] = day
    options["overlap"] = overlap

    X_days, y_days = get_X_y_days(IF_PREP=0, IF_RELOAD=IF_RELOAD)

    X_days = preprocess_X(
        X_days,
        scaler=options["scaler_BL"],
        avg_mean=options["avg_mean_BL"],
        avg_noise=options["avg_noise_BL"],
        unit_var=1,
    )

    model = get_clf(**options)

    options["task"] = "Dual"

    if options["overlap"].lower() == "sample":
        options["features"] = "sample"
        options["task"] = " "
    elif options["overlap"].lower() == "test":
        options["features"] = "test"
    elif options["overlap"].lower() == "reward":
        options["features"] = "reward"
    else:
        options["features"] = "distractor"
        options["task"] = "Dual"

    trials = options["trials"]
    # options['trials'] = 'correct'

    print("getting X, y")
    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    print("X", X_S1_S2.shape, "y", y_S1_S2.shape)

    if options["augment"] == True:
        print("Augment Data with spawner")

        for _ in range(opts["n_aug"]):
            X_aug = spawner(X_S1_S2, y_S1_S2, sigma=options["sig_aug"])
            X_S1_S2 = np.vstack((X_S1_S2, X_aug))
            y_S1_S2 = np.hstack((y_S1_S2, y_S1_S2))

        print("X", X_S1_S2.shape, "y", y_S1_S2.shape)

    if options["overlap"].lower() == "sample":
        X_avg = avg_epochs(X_S1_S2, epochs=options["epoch"])
        sign = -1
    elif options["overlap"].lower() == "test":
        X_avg = avg_epochs(X_S1_S2, epochs=["TEST"])
        sign = -1
    elif options["overlap"].lower() == "reward":
        X_avg = avg_epochs(X_S1_S2, epochs=["RWD2"])
        sign = 1
    else:
        X_avg = avg_epochs(X_S1_S2, epochs=["MD"])
        sign = 1

    # X_avg, y_S1_S2 = SVMSMOTE().fit_resample(X_avg, y_S1_S2)
    print("getting coefficients")
    coefs = get_coefs(model, X_avg, y_S1_S2, **options)

    print(
        "trials", X_S1_S2.shape[0], "coefs", coefs.shape, "non_zero", np.sum(coefs != 0)
    )

    options["task"] = "DualGo"
    options["features"] = features
    options["trials"] = trials

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    print("X", X_S1_S2.shape, "y", y_S1_S2.shape)

    print("computing overlap")
    overlap_A, overlap_B = get_overlap(X_S1_S2, y_S1_S2, coefs)

    # _, overlap_A_ci = my_boots_ci(X_S1_S2,
    #                               lambda X: get_overlap(X, y_S1_S2, coefs, 'A'))

    # _, overlap_B_ci = my_boots_ci(X_S1_S2,
    #                               lambda X: get_overlap(X, y_S1_S2, coefs, 'B'))

    print("bootstrapping")
    _, overlap_ci = my_boots_ci(
        X_S1_S2, lambda X: get_mean_overlap(X, y_S1_S2, coefs, sign)
    )

    # overlap_shuffle = my_shuffle(X_S1_S2, lambda X: get_mean_overlap(X, y_S1_S2, coefs))

    figname = (
        options["overlap"]
        + "_overlap"
        + "_"
        + options["task"]
        + "_"
        + options["trials"]
    )
    fig = plt.figure(figname)

    if options["day"] == "first":
        pal = sns.color_palette("muted")
    if options["day"] == "middle":
        pal = sns.color_palette("pastel")
    if options["day"] == "last":
        pal = sns.color_palette("bright")

    plt.plot(gv.time, overlap_A, "--", color=pal[0])
    plt.plot(gv.time, overlap_B, "--", color=pal[0])

    # plt.fill_between(gv.time, overlap_A-overlap_A_ci[:, 0],
    #                  overlap_A+overlap_A_ci[:, 1], alpha=.25, color=pal[0])

    # plt.fill_between(gv.time, overlap_B-overlap_B_ci[:, 0],
    #                  overlap_B+overlap_B_ci[:, 1], alpha=.25, color=pal[0])

    # plt.plot(time, np.mean(overlap_shuffle, axis=0), '--k')
    plt.plot([0, gv.duration], [0, 0], "--k")

    mean_overlap = (overlap_A + sign * overlap_B) / 2
    # mean_overlap = np.abs(overlap_A - overlap_B)

    plt.plot(gv.time, mean_overlap, color=pal[0])
    plt.fill_between(
        gv.time,
        mean_overlap - overlap_ci[:, 0],
        mean_overlap + overlap_ci[:, 1],
        alpha=0.25,
        color=pal[0],
    )

    add_vlines()
    plt.xlim([0, 12])
    plt.xticks([0, 2, 4, 6, 8, 10, 12])

    plt.xlabel("Time (s)")
    plt.ylabel(options["overlap"] + " overlap")
    plt.yticks([0, 0.01, 0.02])

    pkl_save(fig, figname, path=gv.figdir)
    save_fig(fig, figname, path=gv.figdir)


if __name__ == "__main__":

    plot_overlap(day="first", overlap="sample")
    plot_overlap(day="last", overlap="sample")

    # main(day="first", overlap="reward", features="reward")
    # main(day="last", overlap="reward", features="reward")

    plot_overlap(day="first", overlap="Dist")
    plot_overlap(day="last", overlap="Dist")
