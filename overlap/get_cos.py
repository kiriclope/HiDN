#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from imblearn.over_sampling import SVMSMOTE

import common.constants as gv
from common.options import set_options
from common.plot_utils import save_fig, pkl_save

from data.get_data import get_X_y_days, get_X_y_S1_S2
from preprocess.helpers import avg_epochs, preprocess_X
from preprocess.augmentation import spawner

from decode.classifiers import get_clf
from decode.coefficients import get_coefs

from stats.bootstrap import my_boots_ci
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def circular_convolution(signal, windowSize=10, axis=-1):
    signal_copy = signal.copy()

    if axis != -1 and signal.ndim != 1:
        signal_copy = np.swapaxes(signal_copy, axis, -1)

    ker = np.concatenate(
        (np.ones((windowSize,)), np.zeros((signal_copy.shape[-1] - windowSize,)))
    )
    smooth_signal = np.real(
        np.fft.ifft(
            np.fft.fft(signal_copy, axis=-1) * np.fft.fft(ker, axis=-1), axis=-1
        )
    ) * (1.0 / float(windowSize))

    if axis != 1 and signal.ndim != 1:
        smooth_signal = np.swapaxes(smooth_signal, axis, -1)

    return smooth_signal

def cos_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)

if __name__ == '__main__':

    options = set_options()


    options["features"] = sys.argv[1]
    options["day"] = sys.argv[2]
    task = sys.argv[3]

    try:
        options["day"] = int(options["day"])
    except:
        pass

    X_days, y_days = get_X_y_days(IF_RELOAD=0)

    model = get_clf(**options)

    options["task"] = "Dual"

    options["features"] = "distractor"
    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    X_avg = avg_epochs(X_S1_S2, epochs=["MD"])

    dist_coefs, _ = get_coefs(model, X_avg, y_S1_S2, **options)


    options["features"] = "sample"
    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    X_avg = avg_epochs(X_S1_S2, epochs=["ED"])

    sample_coefs, _ = get_coefs(model, X_avg, y_S1_S2, **options)

    idx_sample = sample_coefs != 0
    idx_dist = dist_coefs != 0

    idx = idx_sample & idx_dist

    idx = np.arange(X_avg.shape[1])

    print('non zeros',  np.sum(idx))

    theta = np.arctan2(dist_coefs[idx], sample_coefs[idx])
    print(theta.shape)

    options["task"] = task

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)

    X_S1_S2 = preprocess_X(
        X_S1_S2,
        scaler=options["scaler_BL"],
        avg_mean=options["avg_mean_BL"],
        avg_noise=options["avg_noise_BL"],
        unit_var=options["unit_var_BL"],
    )

    # X_avg = avg_epochs(X_S1_S2, epochs=["STIM"])

    index_order = theta.argsort()

    # X = X_S1_S2
    X = X_S1_S2[:, idx]
    X = X[:, index_order]
    X = X[y_S1_S2 == 1]
    X_scaled = np.mean(X, 0)
    # X_scaled = X[0]
    print(X_scaled.shape)
    # X_scaled = X[1]
    # X_scaled = StandardScaler().fit_transform(X)
    # X_scaled = MinMaxScaler().fit_transform(X[1])

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        X_scaled,
        interpolation="gaussian",
        origin="lower",
        cmap="jet",
        extent=[0, 14, 0, X_scaled.shape[0]],
        vmin=0.0,
        vmax=1,
        aspect='auto',
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("dF")
    # cbar.set_ticks([-1, 0.25, 0.5, 0.75, 1])
