#!/usr/bin/env python3
import sys
import time
import warnings
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from mne.decoding import (
    GeneralizingEstimator,
    SlidingEstimator,
    cross_val_multiscore,
    get_coef,
)
from sklearn.base import clone
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.utils import resample
from tqdm import tqdm

from src.common.get_data import get_X_y_days, get_X_y_S1_S2
from src.common.options import set_options
from src.common.plot_utils import add_vlines, save_fig
from src.decode.classifiers import get_clf
from src.decode.mne_scores import run_mne_scores
from src.decode.my_mne import my_cross_val_multiscore
from src.preprocess.helpers import avg_epochs, preprocess_X

warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib

golden_ratio = (5**0.5 - 1) / 2
width = 8
matplotlib.rcParams["figure.figsize"] = [width, width * golden_ratio]


def get_ci(res, conf=0.95):
    ostats = np.sort(res, axis=0)
    mean = np.mean(ostats, axis=0)

    p = (1.0 - conf) / 2.0 * 100
    lperc = np.percentile(ostats, p, axis=0)
    lval = mean - lperc

    p = (conf + (1.0 - conf) / 2.0) * 100
    uperc = np.percentile(ostats, p, axis=0)
    uval = -mean + uperc

    ci = np.vstack((lval, uval)).T

    return ci


def get_cv_score(estimator, X, y, cv, n_jobs=-1):
    # calling mne.cross_val_multiscore to compute diagonal score at each time point
    scores = cross_val_multiscore(estimator, X, y, cv=cv, n_jobs=n_jobs, verbose=False)
    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    return scores


def get_shuffle_score(estimator, X, y, cv, n_jobs=-1):
    np.random.seed(None)
    y_copy = y.copy()
    np.random.shuffle(y_copy)

    # calling mne.cross_val_multiscore to compute diagonal score at each time point
    scores = cross_val_multiscore(
        estimator, X, y_copy, cv=cv, n_jobs=n_jobs, verbose=False
    )
    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    return scores


def get_boots_score(estimator, X, y, cv, n_jobs=-1):
    np.random.seed(None)

    X0 = X[y == 0].copy()
    X0 = resample(X0, n_samples=X0.shape[0])

    X1 = X[y == 1].copy()
    X1 = resample(X1, n_samples=X1.shape[0])

    X_boot = np.vstack((X0, X1))

    # calling mne.cross_val_multiscore to compute diagonal score at each time point
    scores = cross_val_multiscore(
        estimator, X_boot, y, cv=cv, n_jobs=n_jobs, verbose=False
    )

    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    return scores


def get_temporal_cv_score(estimator, X, y, cv, n_jobs=-1):
    scores = cross_val_multiscore(estimator, X, y, cv=cv, n_jobs=n_jobs, verbose=False)
    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    return scores


def get_temporal_cv_score_task(estimator, X, X2, y, y2, cv, n_jobs=-1):
    scores = my_cross_val_multiscore(
        estimator, X, X2, y, y2, cv=cv, n_jobs=n_jobs, verbose=False
    )
    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    return scores


def plot_scores_time(scores, ci_scores=None):
    x = np.linspace(0, 14, int(14 * 6))

    figname = "score"
    fig = plt.figure(figname)
    ax = plt.gca()
    plt.plot(x, scores, label="score")
    ax.axhline(0.5, color="k", linestyle="--", label="chance")
    plt.xlabel("Time (s)")
    plt.ylabel("Score")
    add_vlines()
    plt.ylim([0.25, 1])
    plt.yticks([0.25, 0.5, 0.75, 1])

    if ci_scores is not None:
        plt.fill_between(
            x,
            scores - ci_scores[:, 0],
            scores + ci_scores[:, 1],
            alpha=0.2,
            color="k",
        )

    save_fig(fig, figname)


def plot_scores_mat(scores_mat, figname, title):
    fig, ax = plt.subplots(1, 1, num=figname)
    im = ax.imshow(
        scores_mat,
        # interpolation="lanczos",
        origin="lower",
        cmap="jet",
        extent=[0, 14, 0, 14],
        vmin=0.5,
        vmax=1.0,
    )

    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")

    # STIM
    ax.axvline(2, color="w", ls="-", lw=0.5)
    ax.axhline(2, color="w", ls="-", lw=0.5)

    ax.axvline(3, color="w", ls="-", lw=0.5)
    ax.axhline(3, color="w", ls="-", lw=0.5)

    # Dist
    ax.axvline(4.5, color="w", ls="-", lw=0.5)
    ax.axhline(4.5, color="w", ls="-", lw=0.5)

    ax.axvline(5.5, color="w", ls="-", lw=0.5)
    ax.axhline(5.5, color="w", ls="-", lw=0.5)

    # Cue
    ax.axvline(6.5, color="w", ls="-", lw=0.5)
    ax.axhline(6.5, color="w", ls="-", lw=0.5)

    ax.axvline(7, color="w", ls="-", lw=0.5)
    ax.axhline(7, color="w", ls="-", lw=0.5)

    # Test
    ax.axvline(9, color="w", ls="-", lw=0.5)
    ax.axhline(9, color="w", ls="-", lw=0.5)

    ax.axvline(10, color="w", ls="-", lw=0.5)
    ax.axhline(10, color="w", ls="-", lw=0.5)

    plt.xlim([2, 10])
    plt.ylim([2, 10])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Score")
    cbar.set_ticks([0.5, 0.75, 1])

    plt.title(title)
    plt.xticks([2.5, 5.0, 6.75, 9.5], ["Sample", "Dist.", "Cue", "Test"])
    plt.yticks([2.5, 5.0, 6.75, 9.5], ["Sample", "Dist.", "Cue", "Test"])

    save_fig(fig, figname)
    plt.show()


def run_mne_cross_temp(**kwargs):
    options = set_options(**kwargs)

    task = options["task"]

    try:
        options["day"] = int(options["day"])
    except:
        pass
    
    X_days, y_days = get_X_y_days(**options)
    
    model = get_clf(**options)

    # options["task"] = "DPA"
    X, y = get_X_y_S1_S2(X_days, y_days, **options)

    if options['AVG_EPOCHS']:
        X = avg_epochs(X, **options)
    print("X", X.shape, "y", y.shape)
    
    # options["task"] = task
    # X2, y2 = get_X_y_S1_S2(X_days, y_days, **options)

    cv = options["n_out"]

    if options["in_fold"] == "loo":
        cv = LeaveOneOut()

    if options["out_fold"] == "repeated":
        cv = RepeatedStratifiedKFold(
            n_splits=options["n_out"],
            n_repeats=options["n_repeats"],
            random_state=options["random_state"],
        )

    scoring = options["outer_score"]

    # cv = 5
    ci_scores = None

    # cross temporal score
    # define the Temporal generalization object
    estimator = GeneralizingEstimator(model, n_jobs=1, scoring=scoring, verbose=False)

    start_time = time.time()
    scores_mat = get_temporal_cv_score(estimator, X, y, cv, n_jobs=-1)
    
    # scores_mat = get_temporal_cv_score_task(estimator, X, X2, y, y2, cv, n_jobs=-1)
    
    print("--- %s ---" % timedelta(seconds=time.time() - start_time))

    figname = (
        options["mouse"]
        + "_"
        + options["features"]
        + "_cross_temp_score_"
        + options["task"]
        + "_"
        + str(options["day"])
    )

    # title = options["day"] + " " + options["task"]
    # plot_scores_mat(scores_mat, figname, title)

    return scores_mat

if __name__ == "__main__":
    args = sys.argv[1:]  # Exclude the script name from arguments
    options = {k: v for k, v in (arg.split("=") for arg in args)}

    run_mne_cross_temp(**options)
