#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SVMSMOTE

import common.constants as gv
from common.options import set_options
from common.plot_utils import save_fig, pkl_save

from data.get_data import get_X_y_days, get_X_y_S1_S2
from preprocess.helpers import avg_epochs
from preprocess.augmentation import spawner

from decode.classifiers import get_clf
from decode.coefficients import get_coefs

from statistics.bootstrap import my_boots_ci


def cos_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)


def get_cos(X_S1, X_S2, X_D1, X_D2, dist=None, **kwargs):

    options = set_options()

    options["day"] = day
    options["overlap"] = overlap

    X_days, y_days = get_X_y_days(IF_PREP=1, IF_RELOAD=False)

    model = get_clf(**options)

    options["task"] = "Dual"

    if options["overlap"].lower() == "sample":
        options["features"] = "sample"
        options["task"] = " "
    else:
        options["features"] = "distractor"

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    print(X_S1_S2.shape, y_S1_S2.shape)

    if options["overlap"].lower() == "sample":
        X_avg = avg_epochs(X_S1_S2, epochs=["ED"])
    else:  # distractor
        X_avg = avg_epochs(X_S1_S2, epochs=["MD"])

    sample_coefs = get_coefs(model, X_avg, y_S1_S2, **options)
    dist_coefs = get_coefs(model, X_avg, y_S1_S2, **options)

    cosine = np.arccos(cos_between(sample_coefs, dist_coefs)) * 180 / np.pi
