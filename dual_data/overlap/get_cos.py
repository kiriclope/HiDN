import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dual_data.common.get_data import get_X_y_days, get_X_y_S1_S2
from dual_data.common.options import set_options
from dual_data.decode.classifiers import get_clf
from dual_data.decode.coefficients import get_coefs
from dual_data.preprocess.helpers import avg_epochs
from dual_data.overlap.get_overlap import get_coef_feat
from dual_data.decode.bump import circcvl


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
    trials = options['trials']

    try:
        options["day"] = int(options["day"])
    except:
        pass
    
    X_days, y_days = get_X_y_days(**options)
    
    options["features"] = "sample"
    c_sample, _ = get_coef_feat(X_days, y_days, **options)
    print("coefs sample", np.array(c_sample).shape)
    print('non_zeros', np.sum(c_sample>.0001))
        
    options["features"] = "distractor"
    c_dist, _ = get_coef_feat(X_days, y_days, **options)
    print("coefs dist", np.array(c_dist).shape)    
    print('non_zeros', np.sum(c_dist>.0001))
    
    # plt.figure()
    # plt.hist(c_dist, color="b", bins="auto", histtype="step")
    # plt.hist(c_sample, color="r", bins="auto", histtype="step")
    # plt.xlabel("dist")
    
    # theta = np.arctan2(unit_vector(c_dist[idx]), unit_vector(c_sample[idx]))
    # e1 = unit_vector(c_sample)
    # e2 = unit_vector(c_dist)
    # e1, e2 = gram_schmidt(c_sample[idx], c_dist[idx])
    theta = np.arctan2(c_dist, c_sample)
    
    # T = np.column_stack((e1, e2))
    
    index_order = theta.argsort()
    # print(index_order.shape, X.shape)
    
    options['trials'] = trials
    
    X_task = []
    y_task = []

    X_day_task = []
    y_day_task = []
    
    for task in ['DPA', 'DualGo', 'DualNoGo']:
        options["task"] = task    
        
        X, y = get_X_y_S1_S2(X_days, y_days, **options)
        X = X[:, index_order, :]

        X_task.append(X)
        y_task.append(y)
        
        X_day = []
        y_day = []
        
        for day in range(1, options['n_days'] + 1):
            options['day'] = day
            X, y = get_X_y_S1_S2(X_days, y_days, **options)
            X = X[:, index_order, :]
        
            X_day.append(X)
            y_day.append(y)

        X_day_task.append(X)
        y_day_task.append(y)
    
    # vec = np.ones(X.shape[1])
    # new_vec = np.dot(T.T, vec)
    # theta = to_polar_coords_in_N_dims(new_vec)

    # x_t = []
    # for i in range(X.shape[0]):
    #     new_x = T_inv.dot(X[i])
    #     x_t.append(to_polar_coords_in_N_dims(new_x))

    # x_t = np.array(x_t)
    
    # X = X[:, index_order, :]
    # print(X.shape)
    print("Done")
    return X_day_task, y_day_task, X_task, y_task, theta


def plot_bump(X, y, trial, windowSize=10, width=7):
    golden_ratio = (5**.5 - 1) / 2

    fig, ax = plt.subplots(1, 2, figsize= [1.5*width, width * golden_ratio])
    sample = [-1, 1]
    for i in range(2):
        X_sample = X[y == sample[i]].copy()
            
        if windowSize != 0:
            X_scaled = circcvl(X_sample, windowSize, axis=1)
        
        if trial == "all":
            X_scaled = np.mean(X_scaled, 0)
        else:
            X_scaled = X_scaled[trial]
        
        im = ax[i].imshow(
            X_scaled,
            # np.roll(X_scaled, 0, axis=0),
            interpolation="lanczos",
            origin="lower",
            cmap="jet",
            extent=[0, 14, 0, 360],
            # vmin=-2,
            # vmax=2.,
            aspect="auto",
        )
        
        ax[i].set_xlabel("Time (s)")
        ax[i].set_ylabel("Pref. Location (°)")
        ax[i].set_yticks([0, 90, 180, 270, 360])
        ax[i].set_xlim([0, 12])

    cbar = plt.colorbar(im, ax=ax[1])
    cbar.set_label("<Norm. Fluo>")
    cbar.set_ticks([-.5, 0.5, 1.])

if __name__ == "__main__":
    options["features"] = sys.argv[1]
    options["day"] = sys.argv[2]
    options["task"] = sys.argv[3]

    run_get_cos(**options)
