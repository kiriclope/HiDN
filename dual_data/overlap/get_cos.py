import sys

import matplotlib.pyplot as plt
import numpy as np

from dual_data.common.get_data import get_X_y_days, get_X_y_S1_S2
from dual_data.common.options import set_options
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
    e1 = a / np.linalg.norm(a)
    v = b - np.dot(b, e1) * e1      
    # Normalize the vectors (make them unit vectors)
    # e1 = u / np.linalg.norm(u)
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
    trials = options['trials']
    
    try:
        options["day"] = int(options["day"])
    except:
        pass
    
    X_days, y_days = get_X_y_days(**options)
    
    options['trials'] = 'correct'
    options["features"] = "distractor"
    c_dist, _ = get_coef_feat(X_days, y_days, **options)
    print("coefs dist", np.array(c_dist).shape)
    print('non_zeros', np.sum(c_dist>.0001))

    options["features"] = "sample"
    c_sample, _ = get_coef_feat(X_days, y_days, **options)
    print("coefs sample", np.array(c_sample).shape)
    print('non_zeros', np.sum(c_sample>.0001))
        
    # theta = np.arctan2(c_dist, c_sample)
    
    e1, e2 = gram_schmidt(c_sample, c_dist)
    theta = np.arctan2(e2, e1)
    
    index_order = theta.argsort()
    print('idx', index_order.shape, 'c_sample', c_sample.shape)
    
    options['trials'] = trials
    
    X_task = []
    y_task = []

    X_day_task = []
    y_day_task = []
    
    for task in ['DPA', 'DualGo', 'DualNoGo']:
        options["task"] = task  
        
        X, y = get_X_y_S1_S2(X_days, y_days, **options)
        X = X[:, index_order]
        
        X_task.append(X)
        y_task.append(y)
        
        # X_day = []
        # y_day = []
        
        # for day in range(1, options['n_days'] + 1):
        #     options['day'] = day
        #     X, y = get_X_y_S1_S2(X_days, y_days, **options)
        #     X = X[:, index_order, :]
            
        #     X_day.append(X)
        #     y_day.append(y)
        
        # X_day_task.append(X)
        # y_day_task.append(y)
        
    # T = np.column_stack((e1, e2))
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
    # return X_day_task, y_day_task, X_task, y_task, theta
    coefs = np.vstack((c_sample, c_dist))
    print(coefs.shape)
    
    return X_task, y_task, coefs

def plot_bump(X, y, trial, windowSize=10, width=7):
    golden_ratio = (5**.5 - 1) / 2
    
    fig, ax = plt.subplots(1, 2, figsize= [1.5*width, width * golden_ratio])
    sample = [1, -1]
    for i in range(2):
        if i==0:
            rng = np.random.default_rng()
            X_sample = X[y == sample[1]]
            print(X_sample.shape)
            rng.shuffle(X_sample, axis=1)
        else:
            X_sample = X[y == sample[1]]
            
        # print(X_sample.shape)
        
        if windowSize != 0:
            X_scaled = circcvl(X_sample, windowSize, axis=1)
        else:
            X_scaled = X_sample
            
        if trial == "all":
            X_scaled = np.mean(X_scaled, 0)
        else:
            X_scaled = X_scaled[trial]

        
        if i==1:
            im = ax[i].imshow(
                # X_scaled,
                np.roll(X_scaled, int(X_scaled.shape[0]/2), axis=0),
                # interpolation="lanczos",
                origin="lower",
                cmap="jet",
                extent=[0, 14, 0, 360],
                # vmin=-1,
                # vmax=2,
                aspect="auto",  # 
            )
        else:
            im = ax[i].imshow(
                # X_scaled,
                np.roll(X_scaled, int(X_scaled.shape[0]/2), axis=0),
                # interpolation="lanczos",
                origin="lower",
                cmap="jet",
                extent=[0, 14, 0, X_scaled.shape[0]],
                # vmin=-1.5,
                # vmax=2,
                aspect="auto",
            )
            
        if i==0:
            ax[i].set_title('Unordered')
        else:            
            ax[i].set_title('Ordered')
            
        ax[i].set_xlabel("Time (s)")
        if i == 0:
            ax[i].set_ylabel("Neuron #")
        else:
            ax[i].set_ylabel("Pref. Location (°)")            
            ax[i].set_yticks([0, 90, 180, 270, 360])
            
        ax[i].set_xlim([0, 12])

    cbar = plt.colorbar(im, ax=ax[1])
    cbar.set_label("<Norm. Fluo>")
    cbar.set_ticks([-0.5, 0.0, 0.5, 1.0, 1.5])
    
if __name__ == "__main__":
    options["features"] = sys.argv[1]
    options["day"] = sys.argv[2]
    options["task"] = sys.argv[3]

    run_get_cos(**options)
