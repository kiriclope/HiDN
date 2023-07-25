#!/usr/bin/env python3
import sys
import time
from datetime import timedelta

import dual_data.stats.progressbar as pgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dual_data.common.constants import paldict
from dual_data.common.get_data import get_X_y_days, get_X_y_S1_S2
from dual_data.common.options import set_options
from dual_data.common.plot_utils import add_vlines, save_fig
from dual_data.decode.classifiers import get_clf
from dual_data.decode.my_mne import my_cross_val_multiscore
from dual_data.preprocess.helpers import avg_epochs, preprocess_X
from joblib import Parallel, delayed
from mne.decoding import (GeneralizingEstimator, SlidingEstimator,
                          cross_val_multiscore, get_coef)
from sklearn.base import clone
from sklearn.model_selection import (LeaveOneOut, RepeatedStratifiedKFold,
                                     StratifiedKFold)
from sklearn.utils import resample
from tqdm import tqdm


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
    scores = np.nanmean(scores, axis=0)

    return scores


def get_cv_score_task(estimator, X, X2, y, y2, cv, n_jobs=-1):
    # calling mne.cross_val_multiscore to compute diagonal score at each time point
    scores = my_cross_val_multiscore(
        estimator, X, X2, y, y2, cv=cv, n_jobs=n_jobs, verbose=False
    )
    # Mean scores across cross-validation splits
    scores = np.nanmean(scores, axis=0)

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
    scores = np.nanmean(scores, axis=0)

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
    scores = np.nanmean(scores, axis=0)

    return scores


def get_temporal_cv_score(estimator, X, y, cv, n_jobs=-1):
    scores = cross_val_multiscore(estimator, X, y, cv=cv, n_jobs=n_jobs)
    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    return scores


def plot_scores_time(figname, title, scores, ci_scores=None, task="DPA"):
    x = np.linspace(0, 14, scores.shape[0])

    if "first" in figname:
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

    fig = plt.figure(figname)
    ax = plt.gca()
    plt.plot(x, scores, label="score", color=paldict[task])
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
    plt.title(title)
    save_fig(fig, figname)
    plt.show()


def run_mne_scores(**kwargs):
    options = set_options(**kwargs)

    print(options["bootstrap"])

    # options = set_options()
    # options["features"] = sys.argv[1]
    # options["day"] = sys.argv[2]
    # task = sys.argv[3]

    task = options["task"]

    try:
        options["day"] = int(options["day"])
    except:
        pass

    X_days, y_days = get_X_y_days(mouse=options["mouse"], IF_RELOAD=0)

    X_days = preprocess_X(
        X_days,
        scaler=options["scaler_BL"],
        avg_mean=options["avg_mean_BL"],
        avg_noise=options["avg_noise_BL"],
        unit_var=options["unit_var_BL"],
    )

    model = get_clf(**options)

    options["task"] = task
    # options['task'] = 'DPA'
    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    print("X", X.shape, "y", y.shape)

    # options['task'] = task
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

    estimator = SlidingEstimator(model, n_jobs=None, scoring=scoring, verbose=False)

    start_time = time.time()
    scores = get_cv_score(estimator, X, y, cv, n_jobs=-1)

    # scores = get_cv_score_task(estimator, X, X2, y, y2, cv, n_jobs=-1)

    print("--- %s ---" % timedelta(seconds=time.time() - start_time))

    # with pgb.tqdm_joblib(pgb.tqdm(desc="shuffle")):

    #     shuffle_scores = Parallel(n_jobs=-1)(
    #         delayed(get_shuffle_score)(estimator, X, y, cv, n_jobs=None)
    #         for _ in range(10)
    #     )

    # shuffle_scores = np.array(shuffle_scores)

    cv = 5
    ci_scores = None
    if options["bootstrap"] == 1:
        n_boots = options["n_boots"]
        with pgb.tqdm_joblib(pgb.tqdm(desc="bootstrap", total=84 * n_boots)):
            boots_scores = Parallel(n_jobs=-1)(
                delayed(get_boots_score)(estimator, X, y, cv, n_jobs=None)
                for _ in range(n_boots)
            )

        boots_scores = np.array(boots_scores)
        pvalue = (np.sum(boots_scores >= scores, axis=0) + 1.0) / (n_boots + 1)
        ci_scores = get_ci(boots_scores)

    figname = options["mouse"] + "_" + options["features"] + "_score"
    title = options["task"]

    plot_scores_time(figname, title, scores, ci_scores, options["task"])

    # other bootstrap implementation using bagging
    # options["method"] = "bootstrap"
    # options["n_jobs"] = None
    # model = get_clf(**options)
    # scores_boots = get_cv_score(model, X, y, cv, scoring, n_jobs=-1)
    # print(scores_boots.shape)

    # check
    # cross temporal score
    # define the Temporal generalization object
    # estimator = GeneralizingEstimator(model, n_jobs=None,
    # scoring=scoring, verbose=False)
    # scores_mat = get_temporal_cv_score(estimator, X, y, cv, scoring, n_jobs=-1)
    # plot_scores_mat(scores_mat)


if __name__ == "__main__":
    args = sys.argv[1:]  # Exclude the script name from arguments
    options = {k: v for k, v in (arg.split("=") for arg in args)}
    run_mne_scores(**options)
