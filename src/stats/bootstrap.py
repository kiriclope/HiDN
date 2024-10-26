import numpy as np
from joblib import Parallel, delayed
from scipy.stats import bootstrap
from sklearn.utils import resample

from . import progressbar as pgb


def bootstrap_parloop(X, statfunc):
    np.random.seed(None)
    # Sample (with replacement) from the given dataset
    X_sample = resample(X.copy(), n_samples=X.shape[0])
    # Calculate user-defined statistic and store it
    res = statfunc(X_sample)

    return res


def my_boots_ci(X, statfunc, conf=0.95, n_samples=1000, n_jobs=-1, verbose=1):
    """
    Bootstrap the conf intervals for a given sample of a population
    and a statistic.
    Args:
        dataset: A list of values, each a sample from an unknown population
        conf: The conf value (a float between 0 and 1.0)
        iterations: The number of iterations of resampling to perform
        sample_size: The sample size for each of the resampled (0 to 1.0
                     for 0 to 100% of the original data size)
    statistic: The statistic to use. This must be a function that accepts
                   a list of values and returns a single value.
    Returns:
        Returns the upper and lower values of the conf interval.
    """

    if verbose:
        with pgb.tqdm_joblib(
            pgb.tqdm(desc="bootstrap", total=n_samples)
        ) as progress_bar:
            res = Parallel(n_jobs=n_jobs)(
                delayed(bootstrap_parloop)(X, statfunc) for _ in range(n_samples)
            )
    else:
        res = Parallel(n_jobs=n_jobs)(
            delayed(bootstrap_parloop)(X, statfunc) for _ in range(n_samples)
        )

    res = np.asarray(res)
    # print("stats", res.shape)

    # Sort the array of per-sample statistics and cut off ends
    # ostats = sorted(stats)
    try:
        res = res[~np.isnan(res)]
        ostats = np.sort(res, axis=0)
        mean = np.nanmean(ostats, axis=0)

        p = (1.0 - conf) / 2.0 * 100
        lperc = np.nanpercentile(ostats, p, axis=0)
        lval = mean - lperc

        p = (conf + (1.0 - conf) / 2.0) * 100
        uperc = np.nanpercentile(ostats, p, axis=0)
        uval = -mean + uperc
    except:
        lval = np.nan
        uval = np.nan

    ci = np.vstack((lval, uval)).T

    return mean, ci


# def my_boots_ci(X, statfunc, n_samples=10000, method="BCa", alpha=0.05, axis=0):
#     boots_samples = bootstrap(
#         X,
#         statistic=statfunc,
#         n_resamples=n_samples,
#         method=method,
#         confidence_level=1.0 - alpha,
#         vectorized=True,
#         axis=axis,
#     )

#     # print(boots_samples)

#     ci = [boots_samples.confidence_interval.low, boots_samples.confidence_interval.high]
#     mean_boots = np.mean(boots_samples.bootstrap_distribution)

#     ci[0] = mean_boots - ci[0]
#     ci[1] = ci[1] - mean_boots

#     return ci
