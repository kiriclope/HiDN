#!/usr/bin/env python3
from data.get_data import get_X_y_days
from decode.classifiers import get_clf

if __name__ == '__main__':
    X_days, y_days = get_X_y_days(IF_PREP=1)
    clf = get_clf()
