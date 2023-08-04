#!/usr/bin/env python3
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import dual_data.common.constants as gv
from dual_data.common.get_data import get_X_y_days, get_X_y_S1_S2
from dual_data.common.options import set_options
from dual_data.common.plot_utils import add_vlines, save_fig
from dual_data.decode.classifiers import get_clf
from dual_data.decode.coefficients import get_coefs
from dual_data.preprocess.helpers import avg_epochs
from dual_data.stats.bootstrap import my_boots_ci


def proba_to_decision(proba):
    proba = np.hstack(proba)
    return np.log(proba / (1 - proba))


def find_equal_axes(a, b):
    equal_axes = [
        (i, j) for i in range(a.ndim) for j in range(b.ndim) if a.shape[i] == b.shape[j]
    ]
    return equal_axes


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
    coefs, model = get_coefs(model, X_avg, y_S1_S2, **options)

    coefs = coefs / np.linalg.norm(coefs)
    return coefs, model


def projection(x, n, b):
    n_hat = n / np.linalg.norm(n)

    # The projection of x onto the hyperplane
    proj = []
    for i in range(x.shape[0]):
        proj.append(x[i] - (((np.dot(n_hat, x[i]) - b) / np.linalg.norm(n)) * n_hat))

    return np.array(proj)


def get_total_overlap(X, y, eps, coefs, intercept, model, RETURN_AB=0):
    # X = X[:, model.fs_idx_]

    overlap = []
    for i_epoch in range(X.shape[-1]):
        # overlap.append(model.decision_function(X[..., i_epoch]))
        overlap.append(np.dot(coefs, X[..., i_epoch].T))
        # overlap.append(projection(X[..., i_epoch], coefs, intercept))

    overlap = np.array(overlap).T

    idx = np.where(y == 0)[0]
    A = -np.nanmean(overlap[idx], axis=0) / X.shape[1]
    B = -np.nanmean(overlap[~idx], axis=0) / X.shape[1]

    # print("overlap", overlap.shape, "A", A.shape, "B", B.shape)

    if RETURN_AB:
        return A, B, (A + eps * B) / 2
    else:
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

    X_days, y_days = get_X_y_days(mouse=options["mouse"], IF_RELOAD=options["reload"])

    coefs, model = get_coef_feat(X_days, y_days, **options)

    intercept = model.intercept_

    options["task"] = task
    options["features"] = "sample"
    options["trials"] = trials

    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    print("X", X.shape, "y", y.shape)

    A, B, overlap = get_total_overlap(X, y, eps, coefs, intercept, model, RETURN_AB=1)

    overlap_ci = None
    if options["bootstrap"]:
        _, overlap_ci = my_boots_ci(
            X,
            lambda X: get_total_overlap(X, y, -1, coefs, intercept, None),
            n_samples=1000,
        )

    # plot_overlap(data=A, ci=overlap_ci, **options)
    # plot_overlap(data=B, ci=overlap_ci, **options)
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

    plt.plot(gv.time, data, color=paldict[options["task"]])

    plt.plot([0, gv.duration], [0, 0], "--k")

    if ci is not None:
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


def run_all():
    mice = ["ChRM04", "JawsM15", "JawsM18", "ACCM03", "ACCM04"]
    tasks = ["DPA", "DualGo", "DualNoGo"]
    for mouse in mice:
        for task in tasks:
            run_get_overlap(
                mouse=mouse,
                features="distractor",
                task=task,
                day="first",
                method="bolasso",
            )
            run_get_overlap(
                mouse=mouse,
                features="distractor",
                task=task,
                day="last",
                method="bolasso",
            )
            plt.close("all")


if __name__ == "__main__":
    options["features"] = sys.argv[1]
    options["day"] = sys.argv[2]
    options["task"] = sys.argv[3]

    run_get_overlap(**options)
