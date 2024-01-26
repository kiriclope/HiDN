import sys

import matplotlib.pyplot as plt
import numpy as np

from src.common.get_data import get_X_y_days, get_X_y_S1_S2
from src.common.options import set_options
from src.overlap.get_overlap import get_coef_feat
from src.decode.bump import circcvl


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
    options["overlap"] = "ED"
    c_ED, _ = get_coef_feat(X_days, y_days, **options)
    print("coefs ED", np.array(c_ED).shape)
    print('non_zeros', np.sum(c_ED>.0001))

    options["overlap"] = "MD"
    c_MD, _ = get_coef_feat(X_days, y_days, **options)
    print("coefs MD", np.array(c_MD).shape)
    print('non_zeros', np.sum(c_MD>.0001))

    options["overlap"] = "LD"
    c_LD, _ = get_coef_feat(X_days, y_days, **options)
    print("coefs LD", np.array(c_LD).shape)
    print('non_zeros', np.sum(c_LD>.0001))
    
    # options["features"] = "test"
    # c_test, _ = get_coef_feat(X_days, y_days, **options)
    # print("coefs test", np.array(c_test).shape)
    # print('non_zeros', np.sum(c_test>.0001))

    # options["features"] = "distractor"
    # options["overlap"] = "rwd"    
    # c_rwd, _ = get_coef_feat(X_days, y_days, **options)
    # print("coefs rwd", np.array(c_rwd).shape)
    # print('non_zeros', np.sum(c_rwd>.0001))
    
    e1, e2 = gram_schmidt(c_ED, c_dist)
    theta = np.arctan2(e2, e1)
    
    index_order = theta.argsort()
    print('idx', index_order.shape, 'c_sample', c_ED.shape)
    
    options['trials'] = trials
    
    X_task = []
    y_task = []
    
    for task in ['DPA', 'DualGo', 'DualNoGo']:
        options["task"] = task  
        
        X, y = get_X_y_S1_S2(X_days, y_days, **options)
        X = X[:, index_order]
        
        X_task.append(X)
        y_task.append(y)
        
    print("Done")
    coefs = np.vstack((c_ED, c_dist, c_MD, c_LD))
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
