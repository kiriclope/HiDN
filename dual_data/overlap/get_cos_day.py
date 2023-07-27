import matplotlib.pyplot as plt
import numpy as np
from dual_data.common.get_data import get_X_y_days, get_X_y_S1_S2
from dual_data.common.options import set_options
from dual_data.decode.classifiers import get_clf
from dual_data.decode.coefficients import get_coefs
from dual_data.preprocess.helpers import avg_epochs, preprocess_X
from scipy.spatial.distance import cosine


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


def get_coef_feat(X_days, y_days, **options):
    model = get_clf(**options)

    if options["features"] == "sample":
        options["task"] = "Dual"
        options["epoch"] = ["ED"]
    elif options["features"] == "distractor":
        options["task"] = "Dual"
        options["epoch"] = ["MD"]

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    X_avg = avg_epochs(X_S1_S2, epochs=options["epoch"])
    print("X_avg", X_avg.shape)
    coefs, model = get_coefs(model, X_avg, y_S1_S2, **options)

    return coefs


def run_get_cos_day(**kwargs):
    options = set_options(**kwargs)

    X_days, y_days = get_X_y_days(options["mouse"], IF_RELOAD=options["reload"])

    X_days = preprocess_X(
        X_days,
        scaler=options["scaler_BL"],
        avg_mean=options["avg_mean_BL"],
        avg_noise=options["avg_noise_BL"],
        unit_var=options["unit_var_BL"],
    )

    n_days = len(y_days.day.unique())
    days = np.arange(1, n_days + 1)

    cos_day = []
    for day in days:
        options["day"] = day

        options["features"] = "sample"
        c_sample = get_coef_feat(X_days, y_days, **options)
        print("coefs sample", np.array(c_sample).shape)

        options["features"] = "distractor"
        c_dist = get_coef_feat(X_days, y_days, **options)
        print("coefs dist", np.array(c_dist).shape)

        cos = 1 - cosine(c_sample, c_dist)

        cos_day.append(np.arccos(np.clip(cos, -1.0, 1.0)) * 180 / np.pi)

    options["method"] = "bootstrap"
    options["avg_coefs"] = False

    # cos_day = []
    ci_day = []
    for day in days:
        options["day"] = day

        options["features"] = "sample"
        c_sample = get_coef_feat(X_days, y_days, **options)
        print("coefs sample", np.array(c_sample).shape)

        options["features"] = "distractor"
        c_dist = get_coef_feat(X_days, y_days, **options)
        print("coefs dist", np.array(c_dist).shape)

        # cos = 1 - cosine(np.mean(c_sample, 0), np.mean(c_dist, 0))
        # cos_day.append(np.arccos(np.clip(cos, -1.0, 1.0)) * 180 / np.pi)

        cos_boot = []
        for boot in range(c_sample.shape[0]):
            cos = 1 - cosine(c_sample[boot], c_dist[boot])
            cos_boot.append(np.arccos(np.clip(cos, -1.0, 1.0)) * 180 / np.pi)

        cos_boot = np.array(cos_boot)
        print("cos_boot", cos_boot.shape)
        ci_day.append(get_ci(cos_boot)[0])

    ci_day = np.array(ci_day)

    plt.plot(days, cos_day)
    plt.xlabel("Days")
    plt.ylabel("Angle Sample/Dist axes")

    plt.fill_between(
        days,
        cos_day - ci_day[:, 0],
        cos_day + ci_day[:, 1],
        alpha=0.25,
        color="k",
    )

    plt.plot([days[0], days[-1]], [90, 90], "k--")
    plt.show()


if __name__ == "__main__":
    run_get_cos_day()
