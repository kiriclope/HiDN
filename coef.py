#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from common.options import set_options
from data.get_data import get_X_y_days, get_X_y_S1_S2
from preprocess.helpers import avg_epochs
from decode.classifiers import get_clf
from decode.coefficients import get_coefs

def get_overlap(X, coefs):

    overlap = np.zeros((X.shape[-1], X.shape[0]))

    for i_epoch in range(X.shape[-1]):
        overlap[i_epoch] = np.dot(coefs, X[..., i_epoch].T).T
    return -overlap / X.shape[1]

if __name__ == '__main__':
    options = set_options()
    X_days, y_days = get_X_y_days(IF_PREP=1)

    model = get_clf(**options)

    options['task'] = 'Dual'
    options['features'] = 'sample'

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    print(X_S1_S2.shape, y_S1_S2.shape)

    X_avg = avg_epochs(X_S1_S2, epochs=['ED'])

    coefs = get_coefs(model, X_avg, y_S1_S2, **options)

    print("trials", X_S1_S2.shape[0], "coefs", coefs.shape, "non_zero", np.sum(coefs!=0))

    options['task'] = 'DualGo'
    options['features'] = 'sample'

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    print(X_S1_S2.shape, y_S1_S2.shape)

    overlap = get_overlap(X_S1_S2, coefs)

    duration = 14
    frame_rate = 6
    time = np.linspace(0, duration, int(duration*frame_rate))

    idx = np.where(y_S1_S2==0)[0]
    overlap_A = overlap[:, idx]
    overlap_B = overlap[:, ~idx]

    plt.plot(time, np.mean(overlap_A, 1))
    plt.plot(time, np.mean(overlap_B, 1))

    plt.plot(time, (np.mean(overlap_A, 1) + np.mean(overlap_B, 1))/2, 'k')
