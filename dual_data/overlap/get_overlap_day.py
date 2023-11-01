import matplotlib.pyplot as plt
import numpy as np

from dual_data.common.get_data import get_X_y_days, get_X_y_S1_S2
from dual_data.common.options import set_options
from dual_data.preprocess.helpers import avg_epochs
from dual_data.overlap.get_overlap import get_total_overlap, get_coef_feat

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


def run_get_overlap_day(**kwargs):
    options = set_options(**kwargs)
    task = options["task"]
    trials = options["trials"]

    eps = 1
    options["overlap"] = "distractor"
    options["epochs"] = ["MD"]
    if options["features"] == "sample":
        eps = -1
        options["overlap"] = "sample"
        options["epochs"] = ["ED"]
    
    # X_days, y_days = get_X_y_days(options["mouse"], IF_RELOAD=options["reload"])
    X_days, y_days = get_X_y_days(**options)


    n_days = len(y_days.day.unique())
    days = np.arange(1, n_days + 1)
    
    overlap_day = []
    for day in days:
        options["day"] = day
        
        options["features"] = "distractor"
        options["epochs"] = ["MD"]
        coefs, model = get_coef_feat(X_days, y_days, **options)
        print("coefs ", np.array(coefs).shape)

        options["task"] = task
        options["features"] = "sample"
        options["epochs"] = ["ED"]
        options["trials"] = trials
        
        X, y = get_X_y_S1_S2(X_days, y_days, **options)
        X_avg = avg_epochs(X, **options)
        overlap = get_total_overlap(X_avg[..., np.newaxis], y, eps, coefs, 0, model)
        
        print('day', options['day'], 'overlap', overlap)
        overlap_day.append(overlap)

    # options["method"] = "bootstrap"
    # options["avg_coefs"] = False

    # # overlap_day = []
    # ci_day = []
    # for day in days:
    #     options["day"] = day

    #     options["features"] = "sample"
    #     c_sample = get_coef_feat(X_days, y_days, **options)
    #     print("coefs sample", np.array(c_sample).shape)

    #     options["features"] = "distractor"
    #     c_dist = get_coef_feat(X_days, y_days, **options)
    #     print("coefs dist", np.array(c_dist).shape)

    #     # cos = 1 - cosine(np.mean(c_sample, 0), np.mean(c_dist, 0))
    #     # overlap_day.append(np.arccos(np.clip(cos, -1.0, 1.0)) * 180 / np.pi)

    #     cos_boot = []
    #     for boot in range(c_sample.shape[0]):
    #         cos = 1 - cosine(c_sample[boot], c_dist[boot])
    #         cos_boot.append(np.arccos(np.clip(cos, -1.0, 1.0)) * 180 / np.pi)

    #     cos_boot = np.array(cos_boot)
    #     print("cos_boot", cos_boot.shape)
    #     ci_day.append(get_ci(cos_boot)[0])

    # ci_day = np.array(ci_day)
    
    plt.plot(days, overlap_day)
    plt.xlabel("Day")
    plt.ylabel("Dist. Overlap")
    plt.xticks([1, 2, 3, 4 , 5, 6])
    # plt.fill_between(
    #     days,
    #     overlap_day - ci_day[:, 0],
    #     overlap_day + ci_day[:, 1],
    #     alpha=0.25,
    #     color="k",
    # )

    # plt.plot([days[0], days[-1]], [90, 90], "k--")
    
    return overlap_day

if __name__ == "__main__":
    run_get_overlap_day()
