import random

import numpy as np
# import scikits.bootstrap as boot
import scipy.stats as stats
from joblib import Parallel, delayed, parallel_backend
from numpy.random import randint
from sklearn.feature_selection import (SelectKBest, chi2, f_classif,
                                       mutual_info_classif)
from sklearn.utils import resample

from . import progressbar as pgb


def my_F_test(X_S1, X_S2):
    """
    X_Si (n_trials, n_neurons)
    """
    print("X_S1", X_S1.shape, "X_S2", X_S2.shape)

    X_S1_S2 = np.vstack((X_S1, X_S2))

    y = np.concatenate((np.zeros(X_S1.shape[0]), np.ones(X_S2.shape[0])))

    print("X_S1_S2", X_S1_S2.shape, "y_S1_S2", y.shape)

    model = SelectKBest(f_classif, k=X_S1_S2.shape[-1])
    model.fit(X_S1_S2, y)
    pval = model.pvalues_

    print("pval", pval.shape)
    return pval


def shuffle_X_S1_X_S2_parloop(X_S1, X_S2):
    np.random.seed(None)

    # # shuffle
    X_shuffle = np.vstack((X_S1, X_S2))
    np.random.shuffle(X_shuffle)

    X_S1_shuffle = X_shuffle[0 : X_S1.shape[0]]
    X_S2_shuffle = X_shuffle[X_S1.shape[0] :]

    return X_S1_shuffle, X_S2_shuffle


def shuffle_X_S1_X_S2(X_S1, X_S2, n_shuffle=1000, n_jobs=-10):
    with pgb.tqdm_joblib(pgb.tqdm(desc="shuffle", total=n_shuffle)) as progress_bar:
        X_S1_shuffle, X_S2_shuffle = zip(
            *Parallel(n_jobs=n_jobs)(
                delayed(shuffle_X_S1_X_S2_parloop)(X_S1, X_S2) for _ in range(n_shuffle)
            )
        )

    gc.collect()

    return X_S1_shuffle, X_S2_shuffle


def shuffle_parloop(X_S1, X_S2, statfunction):
    np.random.seed(None)

    # # shuffle
    X_shuffle = np.vstack((X_S1, X_S2))
    np.random.shuffle(X_shuffle)

    # pick S1 and S2 trials from shuffle
    X_S1_shuffle = X_shuffle[: X_S1.shape[0]]
    X_S2_shuffle = X_shuffle[X_S1.shape[0] :]

    # apply stat
    sel_shuffle = statfunction(X_S1_shuffle, X_S2_shuffle)

    return sel_shuffle


def shuffle_stat(X_S1, X_S2, statfunction, n_samples=1000, n_jobs=-10):
    with pgb.tqdm_joblib(pgb.tqdm(desc="shuffle", total=n_samples)) as progress_bar:
        sel_shuffle = Parallel(n_jobs=n_jobs)(
            delayed(shuffle_parloop)(X_S1, X_S2, statfunction) for _ in range(n_samples)
        )

    gc.collect()

    sel_shuffle = np.asarray(sel_shuffle)

    return sel_shuffle


def perm_parloop(
    X_S1_DPA, X_S2_DPA, X_S1_other, X_S2_other, i_iter, statfunction, statfunction2
):
    # update random seed
    np.random.seed(None)

    # shuffle S1
    X_shuffle_S1 = np.vstack((X_S1_DPA, X_S1_other))
    np.random.shuffle(X_shuffle_S1)

    X_shuffle_S1_DPA = X_shuffle_S1[: X_S1_DPA.shape[0]]
    X_shuffle_S1_other = X_shuffle_S1[X_S1_DPA.shape[0] :]

    # shuffle S2
    X_shuffle_S2 = np.vstack((X_S2_DPA, X_S2_other))
    np.random.shuffle(X_shuffle_S2)

    X_shuffle_S2_DPA = X_shuffle_S2[: X_S2_DPA.shape[0]]
    X_shuffle_S2_other = X_shuffle_S2[X_S2_DPA.shape[0] :]

    sel_perm = statfunction(X_shuffle_S1_DPA, X_shuffle_S2_DPA)
    sel_perm2 = statfunction2(X_shuffle_S1_other, X_shuffle_S2_other)

    return sel_perm, sel_perm2


def get_sel_perm(
    X_S1_DPA,
    X_S2_DPA,
    X_S1_other,
    X_S2_other,
    statfunction,
    statfunction2,
    n_samples=1000,
    n_jobs=-10,
):
    with pgb.tqdm_joblib(pgb.tqdm(desc="perm", total=n_samples)) as progress_bar:
        sel_perm_DPA, sel_perm_other = zip(
            *Parallel(n_jobs=n_jobs)(
                delayed(perm_parloop)(
                    X_S1_DPA,
                    X_S2_DPA,
                    X_S1_other,
                    X_S2_other,
                    i_iter,
                    statfunction,
                    statfunction2,
                )
                for i_iter in range(n_samples)
            )
        )
    gc.collect()

    sel_perm_DPA = np.asarray(sel_perm_DPA)
    sel_perm_other = np.asarray(sel_perm_other)

    return sel_perm_DPA, sel_perm_other


def bootstrap_parloop(X_S1, X_S2, statfunction):
    np.random.seed(None)
    # Sample (with replacement) from the given dataset
    X_S1_sample = resample(X_S1, n_samples=X_S1.shape[0])
    X_S2_sample = resample(X_S2, n_samples=X_S2.shape[0])

    # Calculate user-defined statistic and store it
    stats = statfunction(X_S1_sample, X_S2_sample)

    return stats


def my_bootstraped_ci(
    X_S1, X_S2, confidence=0.95, n_samples=1000, statfunction=np.mean, n_jobs=-10
):
    """
    Bootstrap the confidence intervals for a given sample of a population
    and a statistic.
    Args:
        dataset: A list of values, each a sample from an unknown population
        confidence: The confidence value (a float between 0 and 1.0)
        iterations: The number of iterations of resampling to perform
        sample_size: The sample size for each of the resampled (0 to 1.0
                     for 0 to 100% of the original data size)
    statistic: The statistic to use. This must be a function that accepts
                   a list of values and returns a single value.
    Returns:
        Returns the upper and lower values of the confidence interval.
    """

    with pgb.tqdm_joblib(pgb.tqdm(desc="bootstrap", total=n_samples)) as progress_bar:
        with parallel_backend("loky", inner_max_num_threads=1):
            stats = Parallel(n_jobs=n_jobs)(  # 10
                delayed(bootstrap_parloop)(X_S1, X_S2, statfunction)
                for _ in range(n_samples)
            )

    # stats = []
    # for _ in pgb.tqdm(range(n_samples), desc='bootstrap') :
    #     stats.append(bootstrap_parloop(X_S1, X_S2, statfunction))

    gc.collect()
    stats = np.asarray(stats)
    print("stats", stats.shape)

    # Sort the array of per-sample statistics and cut off ends
    # ostats = sorted(stats)
    ostats = np.sort(stats, axis=0)
    mean = np.mean(ostats, axis=0)

    # lval = np.percentile(ostats, ((1 - confidence) / 2) * 100, axis=0)
    # uval = np.percentile(ostats, (confidence + ((1 - confidence) / 2)) * 100, axis=0)

    # lval = mean - np.percentile(ostats, ((1 - confidence) / 2) * 100, axis=0)
    # uval = - mean + np.percentile(ostats, (confidence + ((1 - confidence) / 2)) * 100, axis=0)

    p = (1.0 - confidence) / 2.0 * 100
    lperc = np.percentile(ostats, p, axis=0)
    dum = np.vstack((np.zeros(lperc.shape), lperc)).T

    # lval = np.max(dum, axis=1)
    lval = mean - lperc
    # lval = mean - np.max(dum, axis=1)

    p = (confidence + (1.0 - confidence) / 2.0) * 100
    uperc = np.percentile(ostats, p, axis=0)
    dum = np.vstack((np.ones(uperc.shape), uperc)).T

    # uval = np.min( dum, axis=1)
    uval = -mean + uperc
    # uval = -mean + np.min(dum, axis=1)

    # print('mean', mean, 'lower', lperc, 'upper', uperc, 'm-l', lval, 'm+l', uval)

    ci = np.vstack((lval, uval)).T
    # print(ci.shape)

    return mean, ci
