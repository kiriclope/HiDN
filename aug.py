#!/usr/bin/env python3
import numpy as np

from common.options import set_options
from data.get_data import get_X_y_days, get_X_y_S1_S2
from preprocess.helpers import avg_epochs
from preprocess.augmentation import spawner

if __name__ == "__main__":

    options = set_options()

    X_days, y_days = get_X_y_days(IF_PREP=1)

    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    print(X.shape, y.shape)

    X_aug = spawner(X, y)

    print("X_aug", X_aug.shape)
