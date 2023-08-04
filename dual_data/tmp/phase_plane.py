import common.constants as gv
import matplotlib.pyplot as plt
import numpy as np
from common.options import set_options
from common.plot_utils import pkl_save, save_fig
from data.get_data import get_X_y_days, get_X_y_S1_S2
from decode.classifiers import get_clf
from decode.coefficients import get_coefs
from imblearn.over_sampling import SVMSMOTE
from preprocess.augmentation import spawner
from preprocess.helpers import avg_epochs, preprocess_X
from stats.bootstrap import my_boots_ci


def get_overlap(X, y, coefs):

    if coefs.ndim > 1:
        overlap = np.zeros((X.shape[-1], X.shape[0], 4))
    else:
        overlap = np.zeros((X.shape[-1], X.shape[0]))

    print("overlap", overlap.shape)

    for i_epoch in range(X.shape[-1]):
        overlap[i_epoch] = np.dot(coefs, X[..., i_epoch].T).T

    idx = np.where(y == 0)[0]
    overlap_A = -overlap[:, idx] / X.shape[1]
    overlap_B = -overlap[:, ~idx] / X.shape[1]

    overlap = np.stack((overlap_A, overlap_B))
    return overlap
    # return -overlap / X.shape[1]  # normalized by number of neurons


def get_overlap_trials(
    day, overlap, features="sample", task="DualGo", trials="correct", IF_RELOAD=False
):

    options = set_options()

    options["day"] = day
    options["overlap"] = overlap

    X_days, y_days = get_X_y_days(IF_PREP=0, IF_RELOAD=IF_RELOAD)

    X_days = preprocess_X(
        X_days,
        scaler=options["scaler_BL"],
        avg_mean=options["avg_mean_BL"],
        avg_noise=options["avg_noise_BL"],
        unit_var=options["unit_var_BL"],
    )

    model = get_clf(**options)

    options["task"] = " "

    if options["overlap"].lower() == "sample":
        options["features"] = "sample"
    elif options["overlap"].lower() == "distractor":
        options["features"] = "distractor"
        options["task"] = "Dual"
    elif options["overlap"].lower() == "reward":
        options["features"] = "reward"
    elif options["overlap"].lower() == "choice":
        options["features"] = "choice"

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    print(X_S1_S2.shape, y_S1_S2.shape)

    if options["augment"] == True:
        print("Augment Data with spawner")

        for _ in range(options["n_aug"]):
            X_aug = spawner(X_S1_S2, y_S1_S2, sigma=options["sig_aug"])
            X_S1_S2 = np.vstack((X_S1_S2, X_aug))
            y_S1_S2 = np.hstack((y_S1_S2, y_S1_S2))

        print("X", X_S1_S2.shape, "y", y_S1_S2.shape)

    if options["overlap"].lower() == "sample":
        X_avg = avg_epochs(X_S1_S2, epochs=options["epoch_sample"])
    elif options["overlap"].lower() == "distractor":
        X_avg = avg_epochs(X_S1_S2, epochs=options["epoch_dist"])
    elif options["overlap"].lower() == "choice":
        X_avg = avg_epochs(X_S1_S2, epochs=options["epoch_choice"])
    elif options["overlap"].lower() == "reward":
        X_avg = avg_epochs(X_S1_S2, epochs=options["epoch_rwd"])

    options["trials"] = "correct"
    coefs = get_coefs(model, X_avg, y_S1_S2, **options)

    print(
        "trials", X_S1_S2.shape[0], "coefs", coefs.shape, "non_zero", np.sum(coefs != 0)
    )

    options["task"] = task
    options["features"] = features
    options["trials"] = trials

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    print(X_S1_S2.shape, y_S1_S2.shape)

    overlap = get_overlap(X_S1_S2, y_S1_S2, coefs)

    return overlap


def plot_kappa_plane(day="first"):

    overlap_sample = get_overlap_trials(day, overlap="sample")
    overlap_dist = get_overlap_trials(day, overlap="distractor")

    # sample_avg = avg_epochs(overlap_sample.T, epochs=["LD"])
    # dist_avg = avg_epochs(overlap_dist.T, epochs=["LD"])

    # print(sample_avg.shape)
    # print(dist_avg.shape)

    return overlap_sample, overlap_dist

    # radius, theta = carteToPolar(sample_avg, dist_avg)

    # # plt.figure("overlaps_plane_" + day)
    # # plt.plot(np.cos(theta), np.sin(theta), "o")
    # # plt.xlabel("Sample Overlap")
    # # plt.ylabel("Dist. Overlap")

    # plot_phase_dist(day, theta)


def carteToPolar(x, y):
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x) % (2 * np.pi)

    # theta_0 = theta[:, 0]
    # theta_1 = theta[:, 1:]
    # idx = np.diff(theta, axis=1) <= 0

    # theta_1[idx] = np.nan
    # theta = np.hstack((theta_0, theta_1))
    return radius, theta


def plot_phase_dist(day, theta):

    plt.figure("overlaps_phases_" + day)
    plt.hist(theta % 180, histtype="step", density=1, bins="auto")
    plt.xlim([0, 180])
    plt.xticks([0, 45, 90, 135, 180])

    plt.xlabel("Overlaps Pref. Dir. (°)")
    plt.ylabel("Density")


if __name__ == "__main__":

    overlap_sample, overlap_dist = plot_kappa_plane("first")
    # overlap_sample, overlap_dist = plot_kappa_plane("last")
    # plot_kappa_plane("last")
