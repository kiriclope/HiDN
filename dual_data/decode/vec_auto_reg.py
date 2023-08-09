from os import X_OK
import numpy as np

import matplotlib.pyplot as plt

from mne_connectivity import vector_auto_regression

from dual_data.common.options import set_options
from dual_data.common.get_data import get_X_y_days, get_X_y_S1_S2
from dual_data.common.plot_utils import add_vlines
from dual_data.preprocess.helpers import avg_epochs


def circular_convolution(signal, windowSize=10, axis=-1):
    signal_copy = signal.copy()

    if axis != -1 and signal.ndim != 1:
        signal_copy = np.swapaxes(signal_copy, axis, -1)

    ker = np.concatenate(
        (np.ones((windowSize,)), np.zeros((signal_copy.shape[-1] - windowSize,)))
    )
    smooth_signal = np.real(
        np.fft.ifft(
            np.fft.fft(signal_copy, axis=-1) * np.fft.fft(ker, axis=-1), axis=-1
        )
    ) * (1.0 / float(windowSize))

    if axis != -1 and signal.ndim != 1:
        smooth_signal = np.swapaxes(smooth_signal, axis, -1)

    return smooth_signal


def run_VAR(**kwargs):
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
    ch_names = np.arange(0, X.shape[-1])

    conn = vector_auto_regression(data=X, times=times)
    predicted_data = conn.predict(X)

    print(predicted_data.shape)

    # compute residuals
    residuals = X - predicted_data
    residuals = residuals[y == -1]

    X_avg = avg_epochs(residuals, epochs=["ED"])
    print(X_avg.shape)

    idx = np.argsort(np.mean(X_avg, 0))
    print(idx)
    residuals = residuals[:, idx, :]

    # residuals = circular_convolution(residuals, 20, 1)
    print(residuals.shape)

    plt.figure()
    im = plt.imshow(
        np.mean(residuals, 0),
        cmap="jet",
        aspect="auto",
        interpolation="none",
        extent=[0, 14, 0, X.shape[1]],
    )

    # # visualize the residuals
    # fig, ax = plt.subplots()
    # ax.plot(residuals.flatten(), "*")
    # ax.set(title="Residuals of fitted VAR model", ylabel="Magnitude")

    # # compute the covariance of the residuals
    # model_order = conn.attrs.get("model_order")
    # t = residuals.shape[0]
    # sampled_residuals = np.concatenate(
    #     np.split(residuals[:, :, model_order:], t, 0), axis=2
    # ).squeeze(0)
    # rescov = np.cov(sampled_residuals)
    # print(rescov.shape)
    # # Next, we visualize the covariance of residuals.
    # # Here we will see that because we use ordinary
    # # least-squares as an estimation method, the residuals
    # # should come with low covariances.
    # fig, ax = plt.subplots()
    # # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
    # im = ax.imshow(
    #     np.sort(rescov, 0), cmap="jet", aspect="equal", interpolation="lanczos"
    # )
    # fig.colorbar(im, , orientation="horizontal")

    # first_conn = vector_auto_regression(
    #     data=X, times=times, names=ch_names, model="avg-epochs"
    # )

    # first_epoch = X[0, ...]
    # predicted_data = conn.predict(first_epoch)

    # # compute residuals
    # residuals = X - predicted_data

    # # visualize the residuals
    # fig, ax = plt.subplots()
    # ax.plot(residuals.flatten(), "*")
    # ax.set(title="Residuals of fitted VAR model", ylabel="Magnitude")

    # # compute the covariance of the residuals
    # model_order = conn.attrs.get("model_order")
    # t = residuals.shape[0]
    # sampled_residuals = np.concatenate(
    #     np.split(residuals[:, :, model_order:], t, 0), axis=2
    # ).squeeze(0)
    # rescov = np.cov(sampled_residuals)

    # # Next, we visualize the covariance of residuals as before.
    # # Here we will see a similar trend with the covariances as
    # # with the covariances for time-varying VAR model.
    # fig, ax = plt.subplots()
    # # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
    # im = ax.imshow(rescov, cmap="viridis", aspect="equal", interpolation="none")
    # fig.colorbar(im, cax=cax, orientation="horizontal")
