#!/usr/bin/env python3
import numpy as np
from common.options import set_options
from data.get_data import get_X_y_days, get_X_y_S1_S2
from decode.classifiers import get_clf
from decode.methods import outer_temp_cv
from preprocess.helpers import avg_epochs


def get_score(model, X_t_train, X_t_test, y, **options):

    cv_score, pval = outer_temp_cv(model,
                                   X_t_train,
                                   X_t_test,
                                   y,
                                   n_out=options['n_out'],
                                   folds=options['out_fold'],
                                   n_repeats=options['n_repeats'],
                                   outer_score=options['outer_score'],
                                   random_state=options['random_state'],
                                   n_jobs=options['n_jobs'],
                                   IF_SHUFFLE=1)

    print('cv_score', cv_score, 'pval', pval)

    return cv_score


if __name__ == '__main__':
    options = set_options()
    X_days, y_days = get_X_y_days(IF_PREP=1)

    model = get_clf(**options)

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    print(X_S1_S2.shape, y_S1_S2.shape)

    X_S1_S2 = avg_epochs(X_S1_S2, epochs=['ED', 'MD', 'LD'])
    cv_score = get_score(model, X_S1_S2, X_S1_S2, y_S1_S2, **options)
