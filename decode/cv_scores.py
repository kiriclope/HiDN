#!/usr/bin/env python3
import numpy as np

from common.options import set_options
from data.get_data import get_X_y_days, get_X_y_S1_S2
from decode.classifiers import get_clf
from decode.methods import outer_temp_cv

def get_score(model, X_train, X_test, y, **options):

    if X_train.ndim>2:
        cv_score = []
        for i in range(X_train.shape[-1]):
            X_train_i = X_train[..., i]
            X_test_i = X_test[..., i]

            cv_score.append(outer_temp_cv(model, X_train_i, X_test_i, y,
                                          n_out=options['n_out'],
                                          folds=options['out_fold'],
                                          inner_score=options['inner_score'],
                                          outer_score=options['outer_score'],
                                          random_state=None,
                                          n_jobs=options['n_jobs'],
                                          )
                            )

    else:
        cv_score = outer_temp_cv(model, X_train, X_test, y,
                                 n_out=options['n_out'],
                                 folds=options['out_fold'],
                                 inner_score=options['inner_score'],
                                 outer_score=options['outer_score'],
                                 random_state=None,
                                 n_jobs=options['n_jobs'],
                                 )

    return cv_score


if __name__ == '__main__':
    options = set_options()
    X_days, y_days = get_X_y_days(IF_PREP=1, IF_AVG=1)

    model = get_clf(**options)

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    print(X_S1_S2.shape, y_S1_S2.shape)

    cv_score = get_score(model, X_S1_S2, X_S1_S2, y_S1_S2, **options)
    print('cv_score', cv_score)
