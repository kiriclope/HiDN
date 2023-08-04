#!/usr/bin/env python3
import sys
from multiprocessing.context import ForkProcess

import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import seaborn as sns

from dual_data.common.get_data import get_X_y_days, get_X_y_S1_S2
from dual_data.common.options import set_options
from dual_data.decode.classifiers import get_clf
from dual_data.decode.coefficients import get_coefs
from dual_data.overlap.animated_bump import animated_bump
from dual_data.preprocess.helpers import avg_epochs

sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

golden_ratio = (5**0.5 - 1) / 2
width = 7
matplotlib.rcParams["figure.figsize"] = [width, width * golden_ratio]


def to_polar_coords_in_N_dims(x):
    r = np.linalg.norm(x)
    coords = [r]
    for i in range(len(x) - 1):
        # `arccos` returns the angle in radians
        angle = np.arccos(x[i] / np.sqrt(np.sum(x[i:] ** 2)))
        coords.append(angle)
    return np.array(coords)


def gram_schmidt(a, b):
    u = a
    v = b - np.dot(b, u) / np.dot(u, u) * u

    # Normalize the vectors (make them unit vectors)
    e1 = u / np.linalg.norm(u)
    e2 = v / np.linalg.norm(v)

    return e1, e2


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

    if axis != -1 and signal.ndim != 1:
        smooth_signal = np.swapaxes(smooth_signal, axis, -1)

    return smooth_signal


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def cos_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)


def run_get_cos(**kwargs):
    options = set_options(**kwargs)
    task = options["task"]
    try:
        options["day"] = int(options["day"])
    except:
        pass

    X_days, y_days = get_X_y_days(mouse=options["mouse"], IF_RELOAD=options["reload"])

    model = get_clf(**options)

    options["task"] = "Dual"

    options["features"] = "distractor"
    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    X_avg = avg_epochs(X_S1_S2, epochs=["MD"])

    dist_coefs, _ = get_coefs(model, X_avg, y_S1_S2, **options)

    options["task"] = "Dual"

    options["features"] = "sample"
    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    X_avg = avg_epochs(X_S1_S2, epochs=["ED"])

    sample_coefs, _ = get_coefs(model, X_avg, y_S1_S2, **options)

    # plt.figure()
    # plt.hist(dist_coefs, color="b", bins="auto", histtype="step")
    # plt.hist(sample_coefs, color="r", bins="auto", histtype="step")
    # plt.xlabel("dist")

    idx_sample = sample_coefs != 0
    idx_dist = dist_coefs != 0

    idx = idx_sample & idx_dist

    idx = np.arange(X_avg.shape[1])
    print("non zeros", idx.shape)
    # theta = np.arctan2(unit_vector(dist_coefs[idx]), unit_vector(sample_coefs[idx]))

    e1, e2 = gram_schmidt(sample_coefs[idx], dist_coefs[idx])
    theta = np.arctan2(e2, e1)

    # T = np.column_stack((e1, e2))

    options["task"] = task
    X, y = get_X_y_S1_S2(X_days, y_days, **options)

    # vec = np.ones(X.shape[1])
    # new_vec = np.dot(T.T, vec)
    # theta = to_polar_coords_in_N_dims(new_vec)

    # x_t = []
    # for i in range(X.shape[0]):
    #     new_x = T_inv.dot(X[i])
    #     x_t.append(to_polar_coords_in_N_dims(new_x))

    # x_t = np.array(x_t)

    index_order = theta.argsort()
    print(index_order.shape, X.shape)

    X = X[:, index_order, :]
    print(X.shape)
    return X, y


def plot_bump(X, y, sample, trial, windowSize=10):
    X_sample = X[y == sample].copy()

    # X_scaled = StandardScaler().fit_transform(X[trial])
    # X_scaled = MinMaxScaler().fit_transform(X[trial])
    if windowSize != 0:
        X_scaled = circular_convolution(X_sample, windowSize, axis=1)
        # X_scaled = circular_convolution(X_scaled, 8, axis=-1)
        # X_scaled = X_scaled - np.mean(X_scaled, 0)

    else:
        X_scaled = X_sample
        # X_scaled = X - np.mean(X)
        # X_scaled = StandardScaler().fit_transform(X[trial])
        # X_scaled = MinMaxScaler().fit_transform(X[1])

    if trial == "all":
        X_scaled = np.mean(X_scaled, 0)
    else:
        X_scaled = X_scaled[trial]

    # animated_bump(X_scaled)

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        np.roll(X_scaled, 0, axis=0),
        interpolation="lanczos",
        origin="lower",
        cmap="jet",
        extent=[0, 14, 0, 360],
        # vmin=-0.2,
        # vmax=0.2,
        aspect="auto",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("dF")

    plt.xlabel("Time (s)")
    plt.ylabel("Prefered Location (°)")
    plt.yticks([0, 90, 180, 270, 360])
    plt.xlim([0, 12])

    # cbar.set_ticks([-1, 0.25, 0.5, 0.75, 1])

    # # plt.plot(np.mean(X_scaled, 0))
    # trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    # line = plt.Line2D((1, 2), (1.1, 1.1), transform=trans, color="k")
    # ax.add_line(line)

    # fig.subplots_adjust(top=0.8)

    # # This creates a new line from x=0.2 (x=1 in data coords) to x=0.4 (x=2 in data coords) at y=0.9 in figure coords
    # l1 = lines.Line2D(
    #     [4 / 14, 6 / 13],
    #     [0.9, 0.9],
    #     transform=fig.transFigure,
    #     figure=fig,
    # )

    # l2 = lines.Line2D(
    #     [8.5 / 14, 9.5 / 14],
    #     [0.9, 0.9],
    #     transform=fig.transFigure,
    #     figure=fig,
    # )

    # fig.lines.extend([l1])
    # fig.lines.extend([l2])

    # l1 = lines.Line2D(
    #     [6 / 14, 7 / 14], [0.9, 0.9], transform=fig.transFigure, figure=fig
    # )
    # fig.lines.extend([l1])


if __name__ == "__main__":
    options["features"] = sys.argv[1]
    options["day"] = sys.argv[2]
    options["task"] = sys.argv[3]

    run_get_cos(**options)
