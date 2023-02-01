#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import common.constants as gv
from common.options import set_options

from data.get_data import get_X_y_days, get_X_y_S1_S2
from preprocess.helpers import avg_epochs

from decode.classifiers import get_clf
from decode.coefficients import get_coefs

from statistics.bootstrap import my_boots_ci
# from statistics.shuffle import my_shuffle


def add_vlines():
    plt.axvspan(gv.t_STIM[0], gv.t_STIM[1], alpha=0.1, color='b')
    plt.axvspan(gv.t_DIST[0], gv.t_DIST[1], alpha=0.1, color='b')
    plt.axvspan(gv.t_TEST[0], gv.t_TEST[1], alpha=0.1, color='b')
    plt.axvspan(gv.t_CUE[0], gv.t_CUE[1], alpha=0.1, color='g')


def get_overlap(X, coefs):

    overlap = np.zeros((X.shape[-1], X.shape[0]))

    for i_epoch in range(X.shape[-1]):
        overlap[i_epoch] = np.dot(coefs, X[..., i_epoch].T).T
    return -overlap / X.shape[1]


def get_mean_overlap(X, y, coefs):

    overlap = np.zeros((X.shape[-1], X.shape[0]))

    idx = np.where(y == 0)[0]

    for i_epoch in range(X.shape[-1]):
        overlap[i_epoch] = np.dot(coefs, X[..., i_epoch].T).T

    A = np.nanmean(overlap[:, idx], axis=1) / X.shape[1] / 2
    B = np.nanmean(overlap[:, ~idx], axis=1) / X.shape[1] / 2

    return A + B

if __name__ == '__main__':
    options = set_options()
    X_days, y_days = get_X_y_days(IF_PREP=1)

    model = get_clf(**options)

    options['task'] = 'Dual'
    options['features'] = 'distractor'

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    print(X_S1_S2.shape, y_S1_S2.shape)

    X_avg = avg_epochs(X_S1_S2, epochs=['MD'])

    coefs = get_coefs(model, X_avg, y_S1_S2, **options)

    print("trials", X_S1_S2.shape[0], "coefs", coefs.shape,
          "non_zero", np.sum(coefs != 0))

    options['task'] = 'DPA'
    options['features'] = 'sample'

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    print(X_S1_S2.shape, y_S1_S2.shape)

    overlap = get_overlap(X_S1_S2, coefs)
    _, overlap_ci = my_boots_ci(X_S1_S2,
                                lambda X: get_mean_overlap(X, y_S1_S2, coefs))

    # overlap_shuffle = my_shuffle(X_S1_S2, lambda X: get_mean_overlap(X, y_S1_S2, coefs))

    idx = np.where(y_S1_S2 == 0)[0]
    overlap_A = overlap[:, idx]
    overlap_B = overlap[:, ~idx]

    figname = 'overlap' + options['task']
    fig = plt.figure(figname)

    if options['day'] == 'first':
        pal = sns.color_palette('muted')
    else:
        pal = sns.color_palette('bright')

    # plt.plot(gv.time, np.mean(overlap_A, 1), '--')
    # plt.plot(gv.time, np.mean(overlap_B, 1), '--')

    mean_overlap = (np.mean(overlap_A, 1) + np.mean(overlap_B, 1))/2
    plt.plot(gv.time, mean_overlap, color=pal[0])
    # plt.plot(time, np.mean(overlap_shuffle, axis=0), '--k')
    plt.plot([0, gv.duration], [0, 0], '--k')
    plt.fill_between(gv.time, mean_overlap-overlap_ci[:, 0],
                     mean_overlap+overlap_ci[:, 1], alpha=.25, color=pal[0])

    add_vlines()
    plt.xlim([0,14])
    plt.savefig(gv.figdir+figname+'.svg', dpi=300, format='svg')
