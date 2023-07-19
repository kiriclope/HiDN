#!/usr/bin/env python3
import warnings
import sys
import time

from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt

from common.options import set_options
from data.get_data import get_X_y_days, get_X_y_S1_S2
from decode.classifiers import get_clf

from preprocess.helpers import avg_epochs, preprocess_X
import common.plot_utils
from common.plot_utils import add_vlines, save_fig

from sklearn.metrics.pairwise import cosine_similarity

from mne.decoding import (
    SlidingEstimator,
    get_coef,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

def plot_cos_mat(cos_mat, figname, title=None):

    angle = np.arccos(np.clip(cos_mat, -1.0, 1.0)) * 180 / np.pi

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        angle,
        interpolation="lanczos",
        origin="lower",
        cmap="jet",
        extent=[0, 14, 0, 14],
        vmin=0.0,
        vmax=360,
    )

    plt.xlim([2, 12])
    plt.ylim([2, 12])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cos")
    im.set_clim([80, 100])
    plt.xticks([2, 4, 8, 12])
    plt.yticks([2, 4, 8, 12])



if __name__ == "__main__":

    options = set_options()

    options["day"] = sys.argv[1]

    try:
        options["day"] = int(options["day"])
    except:
        pass

    X_days, y_days = get_X_y_days()

    X_days = preprocess_X(
        X_days,
        scaler=options["scaler_BL"],
        avg_mean=options["avg_mean_BL"],
        avg_noise=options["avg_noise_BL"],
        unit_var=options["unit_var_BL"],
    )

    options['method'] = 'bolasso'

    model = get_clf(**options)

    scoring = options["inner_score"]
    estimator = SlidingEstimator(model, n_jobs=-1, scoring=scoring, verbose=False)

    options['features'] = 'sample'
    options['task'] = 'Dual'

    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    print("X", X.shape, "y", y.shape)

    start_time = time.time()
    estimator.fit(X, y)
    coefs_sample = get_coef(estimator, attr='coef_', inverse_transform=False)
    print("--- %s ---" % timedelta(seconds=time.time() - start_time))

    if coefs_sample.shape[1] == 1:
        coefs_sample = coefs_sample[:,0].T
    print('coef sample', coefs_sample.shape)

    options['features'] = 'distractor'
    options['task'] = 'Dual'

    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    print("X", X.shape, "y", y.shape)

    start_time = time.time()
    estimator.fit(X, y)
    coefs_dist = get_coef(estimator, attr='coef_', inverse_transform=False)
    print("--- %s ---" % timedelta(seconds=time.time() - start_time))

    if coefs_dist.shape[1] == 1:
        coefs_dist = coefs_dist[:,0].T
    print('coef dist', coefs_dist.shape)

    # coefs_sample = np.mean(coefs_sample, -1)
    # coefs_dist = np.mean(coefs_dist, -1)
    cos_mat = cosine_similarity(coefs_sample, coefs_dist)

    plot_cos_mat(cos_mat, 'cosine')

    theta = np.arctan2(coefs_dist, coefs_sample)
