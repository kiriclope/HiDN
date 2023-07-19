from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

import common.constants as gv
import common.options

reload(gv)
reload(common.options)

from common.options import set_options
import common.plot_utils
from common.plot_utils import save_fig, pkl_save
from data.get_data import get_X_y_days, get_X_y_S1_S2
from preprocess.helpers import avg_epochs, preprocess_X

from decode.classifiers import get_clf
from decode.coefficients import get_coefs
from stats.bootstrap import my_boots_ci

def get_ci(res, conf=0.95):
    ostats = np.sort(res, axis=0)
    mean = np.mean(ostats, axis=0)

    p = (1.0 - conf) / 2.0 * 100
    lperc = np.percentile(ostats, p, axis=0)
    lval = mean - lperc

    p = (conf + (1.0 - conf) / 2.0) * 100
    uperc = np.percentile(ostats, p, axis=0)
    uval = -mean + uperc

    ci = np.vstack((lval, uval)).T

    return ci

def get_coef_feat(feature, **options):
    model = get_clf(**options)
    
    if feature == "sample":
        options["features"] = "sample"
        options["task"] = "Dual"
        options["epoch"] = ["STIM"]
    else:
        options["features"] = "distractor"        
        options["task"] = "Dual"
        options["epoch"] = ["MD"]
 
    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    X_avg = avg_epochs(X_S1_S2, epochs=options["epoch"])
    
    coefs = get_coefs(model, X_avg, y_S1_S2, **options)

    return coefs    
    
if __name__ == "__main__":

    options = set_options()

    X_days, y_days = get_X_y_days()

    X_days = preprocess_X(
        X_days,
        scaler=options["scaler_BL"],
        avg_mean=options["avg_mean_BL"],
        avg_noise=options["avg_noise_BL"],
        unit_var=1,
    )
    
    options['method'] = 'bolasso'

    coefs_sample = []
    coefs_dist = []

    cos_day = []
    for day in range(1, 7):
        options["day"] = day
        
        c_sample = get_coef_feat('sample', **options) 
        c_dist = get_coef_feat('distractor', **options) 
        
        cos = 1 - cosine(c_sample, c_dist)
        
        cos_day.append(np.arccos(np.clip(cos, -1.0, 1.0)) * 180 / np.pi)
        
        coefs_sample.append(c_sample)
        coefs_dist.append(c_dist)

    options['method'] = 'bootstrap'
    options['avg_coefs'] = False
    
    ci_day = []
    for day in range(1, 7):
        options["day"] = day
        
        c_sample = get_coef_feat('sample', **options) 
        c_dist = get_coef_feat('distractor', **options) 
        print(c_sample.shape)
        
        cos_boot = []
        for boot in range(c_sample.shape[0]):
            cos = 1 - cosine(c_sample[boot], c_dist[boot]) 
            cos_boot.append(np.arccos(np.clip(cos, -1.0, 1.0)) * 180 / np.pi)

        cos_boot = np.array(cos_boot)
        print('cos_boot', cos_boot.shape)
        ci_day.append(get_ci(cos_boot)[0])

    ci_day = np.array(ci_day)
    
    days = np.arange(1,7)
    plt.plot(days, cos_day)
    plt.xlabel("Days")
    plt.ylabel("Angle Sample/Dist axes")

    plt.fill_between(
        days,
        cos_day - ci_day[:, 0],
        cos_day + ci_day[:, 1],
        alpha=0.25,
        color='k',
    )
