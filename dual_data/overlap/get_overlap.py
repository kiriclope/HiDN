import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

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
        options["task"] = "all"
        options["epochs"] = ["ED"]
        options["overlap"] = "sample"
        
    elif options["features"] == "distractor":
        options["task"] = "Dual"
        options["epochs"] = ["MD"]
        options["overlap"] = "distractor"
        
    elif options["features"] == "test":
        options["task"] = "Dual"
        options["epochs"] = ["CHOICE"]
        options["overlap"] = "test"
    
    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    X_avg = avg_epochs(X, **options)
    
    coefs, model = get_coefs(model, X_avg, y, **options)
    
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
        overlap.append(np.dot(coefs, X[..., i_epoch].T) * 100)
        # overlap.append(projection(X[..., i_epoch], coefs, intercept))

    overlap = np.array(overlap).T

    idx = np.where(y == -1)[0]
    A = 0
    if len(idx) > 0:
        A = -np.nanmean(overlap[idx], axis=0) / X.shape[1]
    
    idx = np.where(y == 1)[0]
    B = 0
    if len(idx) > 0:
        B = -np.nanmean(overlap[idx], axis=0) / X.shape[1]
    
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
    
    options['trials'] = 'correct'
    X_days, y_days = get_X_y_days(**options)
    coefs, model = get_coef_feat(X_days, y_days, **options)
    
    # intercept = model.intercept_
    intercept = None
    
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

    if options["show_AB"]:
        plot_overlap(data=A, ci=overlap_ci, **options)
        plot_overlap(data=B, ci=overlap_ci, **options)
    plot_overlap(data=overlap, ci=overlap_ci, **options)


def plot_overlap(data, ci=None, **options):
    figname = (
        options["mouse"] + "_" + options["task"] + "_" + options["overlap"] + "_overlap"
    )

    fig = plt.figure(figname)

    xtime = np.linspace(0, options["duration"], data.shape[-1])
    
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

    plt.plot(xtime, data, color=paldict[options["task"]])
    
    plt.plot([0, options['duration']], [0, 0], "--k")
    
    if ci is not None:
        plt.fill_between(
            xtime,
            data - ci[:, 0],
            data + ci[:, 1],
            alpha=0.25,
            color=paldict[options["task"]],
        )

    add_vlines(mouse=options["mouse"])
    plt.xlim([0, 12])
    plt.xticks([0, 2, 4, 6, 8, 10, 12])
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(4))
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(4))
    
    plt.xlabel("Time (s)")
    plt.ylabel("Overlap")

    save_fig(fig, figname)
    print("Done")


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
