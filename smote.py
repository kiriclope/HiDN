#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import common.constants as gv
from common.options import set_options
from common.plot_utils import save_fig, pkl_save

from data.get_data import get_X_y_days, get_X_y_S1_S2
from preprocess.helpers import avg_epochs

from decode.classifiers import get_clf
from decode.coefficients import get_coefs

from statistics.bootstrap import my_boots_ci

from imblearn.over_sampling import SVMSMOTE

options = set_options()

X_days, y_days = get_X_y_days(IF_PREP=1, IF_RELOAD=0)
X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)

X_S1_S2 = avg_epochs(X_S1_S2, epochs=["ED"])

print(X_S1_S2.shape, y_S1_S2.shape)

X_resampled, y_resampled = SVMSMOTE().fit_resample(X_S1_S2, y_S1_S2)
