import torch
import numpy as np
from time import perf_counter
from src.common.get_data import get_X_y_days, get_X_y_S1_S2
from src.preprocess.helpers import avg_epochs


def convert_seconds(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return h, m, s


def get_classification(model, RETURN="overlaps", **options):
    start = perf_counter()

    dum = 0
    if options["features"] == "distractor":
        if options["task"] != "Dual":
            task = options["task"]
            options["task"] = "Dual"
            dum = 1

    X_days, y_days = get_X_y_days(**options)
    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    y[y == -1] = 0

    X_avg = avg_epochs(X, **options).astype("float32")

    X_test, y_test = None, None

    if dum:
        options["task"] = task
        if "scores" in RETURN:
            options["features"] = "distractor"
            X_test, y_test = get_X_y_S1_S2(X_days, y_days, **options)
        else:
            options["features"] = "sample"
            X, _ = get_X_y_S1_S2(X_days, y_days, **options)

    if options["compo"]:
        print("composition DPA vs", options["compo_task"])
        options["task"] = options["compo_task"]
        X_test, y_test = get_X_y_S1_S2(X_days, y_days, **options)

    if options["verbose"]:
        print("X", X.shape, "y", y.shape)

    # index = mice.index(options['mouse'])
    # model.num_features = N_NEURONS[index]

    # if options["class_weight"]:
    #     pos_weight = torch.tensor(np.sum(y == 0) / np.sum(y == 1), device=DEVICE).to(
    #         torch.float32
    #     )
    #     print("imbalance", pos_weight)
    #     model.criterion__pos_weight = pos_weight

    if RETURN is None:
        return None
    else:
        model.fit(X_avg, y)

    if "scores" in RETURN:
        scores = model.get_cv_scores(
            X, y, options["scoring"], cv=None, X_test=X_test, y_test=y_test
        )
        end = perf_counter()
        print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
        return scores
    elif "overlaps" in RETURN:
        coefs, bias = model.get_bootstrap_coefs(X_avg, y, n_boots=options["n_boots"])
        print("coefs", coefs.shape)
        overlaps = model.get_bootstrap_overlaps(X)
        end = perf_counter()
        print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
        return overlaps
    elif "coefs" in RETURN:
        coefs, bias = model.get_bootstrap_coefs(X_avg, y, n_boots=options["n_boots"])
        end = perf_counter()
        print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
        return coefs, bias
    else:
        return None
