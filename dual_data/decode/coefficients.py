#!/usr/bin/env python3
import numpy as np

from common.options import set_options
from common.get_data import get_X_y_days, get_X_y_S1_S2
from preprocess.helpers import avg_epochs
from decode.classifiers import get_clf


def rescale_coefs(model, coef):
    try:
        scale = model.named_steps["scaler"].scale_
        coefs = np.true_divide(coef, scale)
    except:
        coefs = coef.copy()

    return coefs


def bagged_coefs(model, X, **options):
    if options["multilabel"] or options["multiclass"]:
        coefs = np.zeros((options["n_boots"], 4, X.shape[1]))
    else:
        coefs = np.zeros((options["n_boots"], X.shape[1]))

    # print("coefs", coefs.shape)

    for i_boot in range(options["n_boots"]):
        model_boot = model.estimators_[i_boot]

        try:
            pval = model_boot.named_steps["filter"].pvalues_
            idx = pval <= options["pval"]

            # print("pval", pval.shape)

            if options["multilabel"] or options["multiclass"]:
                coef = model_boot.named_steps["clf"].coef_
                # print("coef", coef.shape)
                coefs[i_boot, :, idx] = coef.T
            else:
                coef = model_boot.named_steps["clf"].coef_[0]
                coefs[i_boot, idx] = coef

        except:
            coef = model_boot.named_steps["clf"].coef_[0]
            coefs[i_boot] = coef

        if options["standardize"] is not None:
            coefs[i_boot] = rescale_coefs(model_boot, coefs[i_boot])

    if options["avg_coefs"]:
        return np.nanmean(coefs, axis=0)
    else:
        return coefs


def get_coefs(model, X, y, **options):
    model.fit(X, y)

    if options["method"] == "gridsearch":
        model = model.best_estimator_

    if options["method"] == "bootstrap":
        coefs = bagged_coefs(model, X, **options)

    elif options["method"] == "bolasso":
        coefs = model.coef_

    elif options["prescreen"]:
        pval = model.named_steps["filter"].pvalues_
        idx = pval <= options["pval"]

        if options["multiclass"] or options["multilabel"]:
            coefs = np.zeros((4, X.shape[1]))
            coef = model.named_steps["clf"].coef_
            # print("coef", coef.shape)

            coefs[:, idx] = coef
        else:
            coefs = np.zeros(X.shape[1])
            coef = model.named_steps["clf"].coef_[0]
            coefs[idx] = coef

        if options["standardize"] is not None:
            coefs = rescale_coefs(model, coefs)

        # print("trials", X.shape[0], "coefs", coefs.shape, "non_zero", coef.shape)
    else:
        coefs = model.named_steps["clf"].coef_[0]

    return coefs, model


if __name__ == "__main__":
    options = set_options()
    X_days, y_days = get_X_y_days(IF_PREP=1)

    model = get_clf(**options)

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)

    X_avg = avg_epochs(X_S1_S2, epochs=["ED"])

    coefs = get_coefs(model, X_avg, y_S1_S2, **options)

    print(
        "trials",
        X_S1_S2.shape[0],
        "coefs",
        coefs.shape,
        "non_zero",
        np.sum(coefs != 0, axis=1),
    )
