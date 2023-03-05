#!/usr/bin/env python3
import time
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt

from common.options import set_options
from data.get_data import get_X_y_days, get_X_y_S1_S2
from decode.classifiers import get_clf

from preprocess.helpers import avg_epochs, preprocess_X
from common.plot_utils import add_vlines

from sklearn.base import clone
from sklearn.utils import resample
from sklearn.model_selection import (
    StratifiedKFold,
    LeaveOneOut,
    RepeatedStratifiedKFold,
)


from mne.decoding import (
    SlidingEstimator,
    GeneralizingEstimator,
    cross_val_multiscore,
    get_coef,
)

from joblib import Parallel, delayed
from tqdm import tqdm
import stats.progressbar as pgb


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
    scores = cross_val_multiscore(estimator, X, y, cv=cv, n_jobs=n_jobs)
    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    return scores


def plot_scores_time(scores, ci_scores=None):
    x = np.linspace(0, 14, int(14 * 6))

    fig = plt.figure("score")
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


def plot_scores_mat(scores_mat):

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        scores_mat,
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=[0, 14, 0, 14],
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")

    # STIM
    ax.axvline(2, color="k", ls="-", lw=0.5)
    ax.axhline(2, color="k", ls="-", lw=0.5)

    # ED
    ax.axvline(3, color="k", ls="-", lw=0.5)
    ax.axhline(3, color="k", ls="-", lw=0.5)

    # ax.axvline(4.5, color="k", ls="--", lw=0.5)
    # ax.axhline(4.5, color="k", ls="--", lw=0.5)

    # Dist
    ax.axvline(4.5, color="k", ls="-", lw=0.5)
    ax.axhline(4.5, color="k", ls="-", lw=0.5)

    ax.axvline(5.5, color="k", ls="-", lw=0.5)
    ax.axhline(5.5, color="k", ls="-", lw=0.5)

    # MD
    # ax.axvline(5.5, color="k", ls="--", lw=0.5)
    # ax.axhline(5.5, color="k", ls="--", lw=0.5)

    # ax.axvline(6.5, color="k", ls="--", lw=0.5)
    # ax.axhline(6.5, color="k", ls="--", lw=0.5)

    # LD
    # ax.axvline(7.5, color="k", ls="--", lw=0.5)
    # ax.axhline(7.5, color="k", ls="--", lw=0.5)

    # ax.axvline(9, color="k", ls="--", lw=0.5)
    # ax.axhline(9, color="k", ls="--", lw=0.5)

    # Test
    ax.axvline(9, color="k", ls="-", lw=0.5)
    ax.axhline(9, color="k", ls="-", lw=0.5)

    ax.axvline(10, color="k", ls="-", lw=0.5)
    ax.axhline(10, color="k", ls="-", lw=0.5)

    plt.xlim([0, 12])
    plt.ylim([0, 12])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Score")


if __name__ == "__main__":
    options = set_options()

    X_days, y_days = get_X_y_days()

    X_days = preprocess_X(
        X_days,
        scaler=options["scaler_BL"],
        avg_mean=options["avg_mean_BL"],
        avg_noise=options["avg_noise_BL"],
        unit_var=options["unit_var_BL"],
    )

    model = get_clf(**options)

    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    print("X", X.shape, "y", y.shape)

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
    print("--- %s ---" % timedelta(seconds=time.time() - start_time))

    # with pgb.tqdm_joblib(pgb.tqdm(desc="shuffle")):

    #     shuffle_scores = Parallel(n_jobs=-1)(
    #         delayed(get_shuffle_score)(estimator, X, y, cv, n_jobs=None)
    #         for _ in range(10)
    #     )

    # shuffle_scores = np.array(shuffle_scores)

    cv = 5
    ci_scores = None
    if options["bootstrap"]:
        n_boots = 100
        with pgb.tqdm_joblib(pgb.tqdm(desc="bootstrap", total=84 * n_boots)):

            boots_scores = Parallel(n_jobs=-1)(
                delayed(get_boots_score)(estimator, X, y, cv, n_jobs=None)
                for _ in range(n_boots)
            )

            boots_scores = np.array(boots_scores)
            pvalue = (np.sum(boots_scores >= scores, axis=0) + 1.0) / (n_boots + 1)
            ci_scores = get_ci(boots_scores)

    plot_scores_time(scores, ci_scores)

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
