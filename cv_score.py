#!/usr/bin/env python3
import numpy as np
from imblearn.over_sampling import SVMSMOTE

from dask.distributed import Client
from dask.distributed import LocalCluster

from common.options import set_options
from data.get_data import get_X_y_days, get_X_y_S1_S2
from preprocess.helpers import avg_epochs, preprocess_X
from decode.classifiers import get_clf
from decode.methods import outer_temp_cv


def augment_imbal(X, y):

    if X.ndim > 2:
        X_res = []
        y_res = []
        for i in range(X.shape[-1]):
            X_i, y_i = SVMSMOTE().fit_resample(X[..., i], y)
            X_res.append(X_i)
            y_res.append(y_i)
        X_res = np.array(X_res)
        y_res = np.array(y_res)
    else:
        X_res, y_res = SVMSMOTE().fit_resample(X, y)

    print("X_res", X_res[0].shape, "y_res", y_res[0].shape)
    print("X_res", X_res[1].shape, "y_res", y_res[1].shape)
    return X_res, y_res


def get_score(model, X_t_train, X_t_test, y, **options):

    scores, perm_scores, pvals = outer_temp_cv(
        model,
        X_t_train,
        X_t_test,
        y,
        n_out=options["n_out"],
        folds=options["out_fold"],
        n_repeats=options["n_repeats"],
        outer_score=options["outer_score"],
        random_state=options["random_state"],
        n_jobs=options["n_jobs"],
        IF_SHUFFLE=options["shuffle"],
    )

    print(
        "scores", scores.shape, "perm_scores", perm_scores.shape, "pvals", pvals.shape
    )

    return scores, perm_scores, pvals


if __name__ == "__main__":

    cluster = LocalCluster()
    client = Client(cluster)

    options = set_options()
    X_days, y_days = get_X_y_days()

    X_days = preprocess_X(
        X_days,
        scaler=options["scaler_BL"],
        avg_mean=options["avg_mean_BL"],
        avg_noise=options["avg_noise_BL"],
        unit_var=options["unit_var_BL"],
    )

    options["n_jobs"] = -1
    model = get_clf(**options)

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    print(X_S1_S2.shape, y_S1_S2.shape)

    X_S1_S2 = avg_epochs(X_S1_S2, epochs=["ED", "MD", "LD"])
    scores, perm_scores, pvals = get_score(model, X_S1_S2, X_S1_S2, y_S1_S2, **options)

    cluster.close()
    client.close()
