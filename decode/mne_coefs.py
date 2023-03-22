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

    options["features"] = sys.argv[1]
    options["day"] = sys.argv[2]
    options["task"] = sys.argv[3]

    X_days, y_days = get_X_y_days()

    X_days = preprocess_X(
        X_days,
        scaler=options["scaler_BL"],
        avg_mean=options["avg_mean_BL"],
        avg_noise=options["avg_noise_BL"],
        unit_var=options["unit_var_BL"],
    )

    options['method'] = 'bootstrap'
    
    model = get_clf(**options)

    scoring = options["inner_score"]
    estimator = SlidingEstimator(model, n_jobs=-1, scoring=scoring, verbose=False)
    
    options['features'] = 'sample'
    options['task'] = 'Dual'    

    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    print("X", X.shape, "y", y.shape)
    
    start_time = time.time()
    estimator.fit(X, y)
    coefs = get_coef(estimator, attr='coef_', inverse_transform=False)
    print("--- %s ---" % timedelta(seconds=time.time() - start_time))

    coefs_sample = coefs[:,0].T

    options['features'] = 'distractor'
    options['task'] = 'Dual'

    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    print("X", X.shape, "y", y.shape)
        
    start_time = time.time()
    estimator.fit(X, y)
    coefs = get_coef(estimator, attr='coef_', inverse_transform=False)
    print("--- %s ---" % timedelta(seconds=time.time() - start_time))

    coefs_dist = coefs[:,0].T
 
    print(coefs.shape)
    cos_mat = cosine_similarity(coefs_sample, coefs_dist)

    plot_cos_mat(cos_mat, 'cosine')
