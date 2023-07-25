#!/usr/bin/env python3
import sys
import time
from datetime import timedelta

import dual_data.stats.progressbar as pgb
import matplotlib.pyplot as plt
import numpy as np
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
    x = np.arange(1, 7)

    fig = plt.figure(figname)
    ax = plt.gca()
    plt.plot(x, scores, label="score", color=paldict[task])
    ax.axhline(0.5, color="k", linestyle="--", label="chance")
    plt.xlabel("Day")
    plt.ylabel("Score")
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


def run_scores_days(**kwargs):
    options = set_options(**kwargs)

    X_days, y_days = get_X_y_days(mouse=options["mouse"], IF_RELOAD=0)

    X_days = preprocess_X(
        X_days,
        scaler=options["scaler_BL"],
        avg_mean=options["avg_mean_BL"],
        avg_noise=options["avg_noise_BL"],
        unit_var=options["unit_var_BL"],
    )

    model = get_clf(**options)

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

    # estimator = SlidingEstimator(model, n_jobs=None, scoring=scoring, verbose=False)
    estimator = model
    start_time = time.time()

    # if options['features'] == 'paired':
    epoch = "RWD2"
    # if options['features'] == 'paired':

    scores_day = []
    for i_day in range(1, 7):
        options["day"] = i_day
        X, y = get_X_y_S1_S2(X_days, y_days, **options)
        X_avg = avg_epochs(X, epochs=[epoch])

        print("day", i_day, "X", X_avg.shape, "y", y.shape)

        scores = get_cv_score(estimator, X_avg, y, cv, n_jobs=-1)

        print(scores)
        scores_day.append(scores)

    print("--- %s ---" % timedelta(seconds=time.time() - start_time))

    figname = options["features"] + "cross_temp_scores_day_"

    title = options["task"]
    ci_scores = None
    plot_scores_time(figname, title, scores_day, ci_scores, options["task"])


if __name__ == "__main__":
    options["features"] = sys.argv[1]
    options["day"] = sys.argv[2]
    options["task"] = sys.argv[3]

    run_scores_days(**options)
