import sys
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io, EvokedArray
from mne.datasets import sample
from mne.decoding import EMS, compute_ems
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

from src.common.options import set_options
from src.common.get_data import get_X_y_days, get_X_y_S1_S2
from src.common.plot_utils import add_vlines


def run_ems(**kwargs):
    options = set_options(**kwargs)
    task = options["task"]

    try:
        options["day"] = int(options["day"])
    except ValueError:
        pass

    # X_days, y_days = get_X_y_days(mouse=options["mouse"], IF_RELOAD=options["reload"])
    X_days, y_days = get_X_y_days(**options)

    options["task"] = task
    # options['task'] = 'DPA'
    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    print("X", X.shape, "y", y.shape)

    n_epochs, n_channels, n_times = X.shape
    times = np.linspace(0, 14, n_times)

    # Initialize EMS transformer
    ems = EMS()

    # Initialize the variables of interest
    X_transform = np.zeros((n_epochs, n_times))  # Data after EMS transformation
    filters = list()  # Spatial filters at each time point

    # In the original paper, the cross-validation is a leave-one-out. However,
    # we recommend using a Stratified KFold, because leave-one-out tends
    # to overfit and cannot be used to estimate the variance of the
    # prediction within a given fold.
    cv = LeaveOneOut()

    for train, test in cv.split(X, y):
        # In the original paper, the z-scoring is applied outside the CV.
        # However, we recommend to apply this preprocessing inside the CV.
        # Note that such scaling should be done separately for each channels if the
        # data contains multiple channel types.
        X_scaled = (X - np.mean(X[train])) / np.std(X[train])

        # Fit and store the spatial filters
        ems.fit(X_scaled[train], y[train])

        # Store filters for future plotting
        filters.append(ems.filters_)

        # Generate the transformed data
        X_transform[test] = ems.transform(X_scaled[test])

    # Average the spatial filters across folds
    filters = np.mean(filters, axis=0)

    # Plot individual trials
    plt.figure("single trials")
    plt.title("single trial surrogates")
    plt.imshow(
        X_transform[y.argsort()],
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], 1, len(X_transform)],
        cmap="jet",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Trials (reordered by condition)")

    # Plot average response
    plt.figure("Average EMS signal")
    plt.title("Average EMS signal")

    ems_ave_1 = X_transform[y == -1]
    # plt.plot(times, ems_ave.mean(0), label="S1")

    ems_ave_2 = X_transform[y == 1]
    plt.plot(times, ems_ave_1.mean(0) - ems_ave_2.mean(0))
    add_vlines()

    plt.xlabel("Time (s)")
    plt.ylabel("a.u.")
    # plt.legend(loc="best")
    plt.show()

    # # Visualize spatial filters across time
    # evoked = EvokedArray(filters, epochs.info, tmin=epochs.tmin)
    # evoked.plot_topomap(scalings=1)


if __name__ == "__main__":
    args = sys.argv[1:]  # Exclude the script name from arguments
    options = {k: v for k, v in (arg.split("=") for arg in args)}
    run_ems(**options)
