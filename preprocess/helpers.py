import numpy as np
from scipy.stats import norm, circmean

# from scipy.signal import savgol_filter, detrend

from common import constants as gv


def center_BL(X, center=None, avg_mean=0):

    if center is None:
        X_BL = X[..., gv.bins_BL]

        if avg_mean:
            print("avg mean over trials")
            center = np.mean(np.hstack(X_BL), axis=-1)
        else:
            center = np.nanmean(X_BL, axis=-1)

    if avg_mean:
        X_scale = X - center[np.newaxis, ..., np.newaxis]
    else:
        X_scale = X - center[..., np.newaxis]

    return X_scale


def standard_scaler_BL(X, center=None, scale=None, avg_mean=0, avg_noise=0):

    X_BL = X[..., gv.bins_BL]
    if center is None:
        if avg_mean:
            print("avg mean over trials")
            center = np.mean(np.hstack(X_BL), axis=-1)
        else:
            center = np.nanmean(X_BL, axis=-1)

    if scale is None:
        if avg_noise:
            print("avg noise over trials")
            scale = np.nanstd(np.hstack(X_BL), axis=-1)
        else:
            scale = np.nanstd(X_BL, axis=-1)

        scale = _handle_zeros_in_scale(scale, copy=False)

    if avg_mean and avg_noise:
        X_scale = (X - center[np.newaxis, ..., np.newaxis]) / scale[
            np.newaxis, ..., np.newaxis
        ]
    elif avg_mean:
        X_scale = (X - center[np.newaxis, ..., np.newaxis]) / scale[..., np.newaxis]
    elif avg_noise:
        X_scale = (X - center[..., np.newaxis]) / scale[np.newaxis, ..., np.newaxis]
    else:
        X_scale = (X - center[..., np.newaxis]) / scale[..., np.newaxis]

    return X_scale, center, scale


def robust_scaler_BL(X, center=None, scale=None, avg_mean=0, avg_noise=0, unit_var=0):

    X_BL = X[..., gv.bins_BL]

    if center is None:
        if avg_mean:
            print("avg mean over trials")
            center = np.nanmedian(np.hstack(X_BL), axis=-1)
        else:
            center = np.nanmedian(X_BL, axis=-1)

    if scale is None:
        if avg_noise:
            print("avg noise over trials")
            quantiles = np.nanpercentile(np.hstack(X_BL), q=[25, 75], axis=-1)
        else:
            quantiles = np.nanpercentile(X_BL, q=[25, 75], axis=-1)

        scale = quantiles[1] - quantiles[0]
        scale = _handle_zeros_in_scale(scale, copy=False)

        if unit_var:
            adjust = norm.ppf(75 / 100.0) - norm.ppf(25 / 100.0)
            scale = scale / adjust

    if avg_mean and avg_noise:
        X_scale = (X - center[np.newaxis, ..., np.newaxis]) / scale[
            np.newaxis, ..., np.newaxis
        ]
    elif avg_mean:
        X_scale = (X - center[np.newaxis, ..., np.newaxis]) / scale[..., np.newaxis]
    elif avg_noise:
        X_scale = (X - center[..., np.newaxis]) / scale[np.newaxis, ..., np.newaxis]
    else:
        X_scale = (X - center[..., np.newaxis]) / scale[..., np.newaxis]

    return X_scale


def _handle_zeros_in_scale(scale, copy=True):
    """Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.
    """

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


def df_f_scaler_BL(X, center=None, avg_mean=0):
    X_BL = X[..., gv.bins_BL]

    if center is None:
        if avg_mean:
            print("avg mean over trials")
            center = np.mean(np.hstack(X_BL), axis=-1)
        else:
            center = np.nanmean(X_BL, axis=-1)

    center = _handle_zeros_in_scale(center, copy=False)

    if avg_mean:
        X_scale = (X - center[np.newaxis, ..., np.newaxis]) / center[..., np.newaxis]
    else:
        X_scale = (X - center[..., np.newaxis]) / center[..., np.newaxis]

    return X_scale, center


def preprocess_X_S1_X_S2(
    X_S1,
    X_S2,
    scaler="standard",
    center=None,
    scale=None,
    avg_mean=0,
    avg_noise=0,
    unit_var=0,
    return_center_scale=0,
    same=1,
):

    X = np.vstack((X_S1, X_S2))
    # X = savgol_filter(X, int(np.ceil(gv.frame_rate/2.0) * 2 + 1), polyorder = gv.SAVGOL_ORDER, deriv=0, axis=-1, mode='mirror')

    if scaler == "standard":
        X_scale, center, scale = standard_scaler_BL(
            X, center, scale, avg_mean, avg_noise
        )
    elif scaler == "robust":
        X_scale = robust_scaler_BL(X, center, scale, avg_mean, avg_noise, unit_var)
    elif scaler == "center":
        X_scale = center_BL(X, center, avg_mean)
    elif scaler == "dff":
        X_scale, center = df_f_scaler_BL(X, center, avg_mean)
    else:
        X_scale = X

    # X_scale = savgol_filter(X_scale, int(np.ceil(gv.frame_rate/2.0) * 2 + 1), polyorder = gv.SAVGOL_ORDER, deriv=0, axis=-1, mode='mirror')

    # X_scale = detrend(X_scale, bp=[gv.bins_STIM[0], gv.bins_DIST[0]])

    if return_center_scale:
        return X_scale[: X_S1.shape[0]], X_scale[X_S1.shape[0] :], center, scale
    else:
        return X_scale[: X_S1.shape[0]], X_scale[X_S1.shape[0] :]


def preprocess_X(
    X,
    scaler="standard",
    center=None,
    scale=None,
    avg_mean=0,
    avg_noise=0,
    unit_var=0,
    return_center_scale=0,
):

    # X = savgol_filter(X, int(np.ceil(gv.frame_rate/2.0) * 2 + 1), polyorder = gv.SAVGOL_ORDER, deriv=0, axis=-1, mode='mirror')

    if scaler == "standard":
        X_scale, center, scale = standard_scaler_BL(
            X, center, scale, avg_mean, avg_noise
        )
    elif scaler == "robust":
        X_scale = robust_scaler_BL(X, center, scale, avg_mean, avg_noise, unit_var)
    elif scaler == "center":
        X_scale = center_BL(X, center, avg_mean)
    elif scaler == "dff":
        X_scale, center = df_f_scaler_BL(X, center, avg_mean)
    else:
        X_scale = X

    # X_scale = savgol_filter(X_scale, int(np.ceil(gv.frame_rate/2.0) * 2 + 1), polyorder = gv.SAVGOL_ORDER, deriv=0, axis=-1, mode='mirror')

    # X_scale = detrend(X_scale, bp=[gv.bins_STIM[0], gv.bins_DIST[0]])

    if return_center_scale:
        return X_scale, center, scale
    else:
        return X_scale


def avg_epochs(X, epochs=None):

    X_avg = np.nanmean(X, axis=-1)

    if epochs is None:
        epochs = gv.epochs

    X_epochs = np.empty(tuple([len(epochs)]) + X_avg.shape)
    # print('X', X_epochs.shape, 'X_avg', X_avg.shape)
    # print('start', gv.bin_start, 'epochs', gv.epochs)
    # print('average over epochs', epochs)

    for i_epoch, epoch in enumerate(epochs):

        if epoch == "BL":
            X_BL = np.nanmean(X[..., gv.bins_BL], axis=-1)
            X_epochs[i_epoch] = X_BL
        elif epoch == "STIM":
            X_STIM = np.nanmean(X[..., gv.bins_STIM], axis=-1)
            X_epochs[i_epoch] = X_STIM
        elif epoch == "STIM_ED":
            X_STIM_ED = np.nanmean(X[..., gv.bins_STIM + gv.bins_ED], axis=-1)
            X_epochs[i_epoch] = X_STIM_ED
        elif epoch == "ED":
            X_ED = np.nanmean(X[..., gv.bins_ED], axis=-1)
            # print('X_ED', X_ED.shape, 'bins', gv.bins_ED)
            X_epochs[i_epoch] = X_ED
        elif epoch == "DIST":
            X_DIST = np.nanmean(X[..., gv.bins_DIST], axis=-1)
            X_epochs[i_epoch] = X_DIST
        elif epoch == "MD":
            X_MD = np.nanmean(X[..., gv.bins_MD], axis=-1)
            X_epochs[i_epoch] = X_MD
        elif epoch == "CUE":
            X_CUE = np.nanmean(X[..., gv.bins_CUE], axis=-1)
            X_epochs[i_epoch] = X_CUE
        elif epoch == "LD":
            X_LD = np.nanmean(X[..., gv.bins_LD], axis=-1)
            X_epochs[i_epoch] = X_LD
        elif epoch == "RWD":
            X_RWD = np.nanmean(X[..., gv.bins_RWD], axis=-1)
            X_epochs[i_epoch] = X_RWD
        elif epoch == "TEST":
            X_TEST = np.nanmean(X[..., gv.bins_TEST], axis=-1)
            X_epochs[i_epoch] = X_TEST
        elif epoch == "DELAY":
            X_DELAY = np.nanmean(X[..., gv.bins_DELAY], axis=-1)
            X_epochs[i_epoch] = X_DELAY
        elif epoch == "Before":
            X_bef = np.nanmean(X[..., gv.time < gv.t_DIST[0]], axis=-1)
            print(X_bef.shape)
            X_epochs[i_epoch] = X_bef
        elif epoch == "After":
            X_bef = np.nanmean(X[..., np.where(gv.time > gv.t_DIST[1])], axis=-1)
            X_epochs[i_epoch] = X_bef
        elif epoch == "RWD2":
            X_RWD = np.nanmean(X[..., gv.bins_RWD2], axis=-1)
            X_epochs[i_epoch] = X_RWD

    X_epochs = np.moveaxis(X_epochs, 0, -1)
    # print("X_epochs", X_epochs.shape, epochs)

    if X_epochs.shape[-1] == 1:
        X_epochs = X_epochs[..., 0]

    return X_epochs


def avg_phase_epochs(X, epochs=None):

    X_avg = np.nanmean(X, axis=-1)

    if epochs is None:
        epochs = gv.epochs

    X_epochs = np.empty(tuple([len(epochs)]) + X_avg.shape)
    # print('X', X_epochs.shape, 'X_avg', X_avg.shape)
    # print('start', gv.bin_start, 'epochs', gv.epochs)
    # print('average over epochs', epochs)

    for i_epoch, epoch in enumerate(epochs):

        if epoch == "BL":
            X_BL = circmean(X[..., gv.bins_BL], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_BL
        elif epoch == "STIM":
            X_STIM = circmean(X[..., gv.bins_STIM], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_STIM
        elif epoch == "STIM_ED":
            X_STIM_ED = circmean(
                X[..., gv.bins_STIM + gv.bins_ED], axis=-1, high=360, low=0
            )
            X_epochs[i_epoch] = X_STIM_ED
        elif epoch == "ED":
            X_ED = circmean(X[..., gv.bins_ED], axis=-1, high=360, low=0)
            # print('X_ED', X_ED.shape, 'bins', gv.bins_ED)
            X_epochs[i_epoch] = X_ED
        elif epoch == "DIST":
            X_DIST = circmean(X[..., gv.bins_DIST], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_DIST
        elif epoch == "MD":
            X_MD = circmean(X[..., gv.bins_MD], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_MD
        elif epoch == "CUE":
            X_CUE = circmean(X[..., gv.bins_CUE], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_CUE
        elif epoch == "LD":
            X_LD = circmean(X[..., gv.bins_LD], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_LD
        elif epoch == "RWD":
            X_RWD = circmean(X[..., gv.bins_RWD], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_RWD
        elif epoch == "TEST":
            X_TEST = circmean(X[..., gv.bins_TEST], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_TEST
        elif epoch == "DELAY":
            X_DELAY = circmean(X[..., gv.bins_DELAY], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_DELAY
        elif epoch == "Before":
            X_bef = circmean(X[..., gv.time < gv.t_DIST[0]], axis=-1, high=360, low=0)
            print(X_bef.shape)
            X_epochs[i_epoch] = X_bef
        elif epoch == "After":
            X_bef = circmean(
                X[..., np.where(gv.time > gv.t_DIST[1])], axis=-1, high=360, low=0
            )
            X_epochs[i_epoch] = X_bef
        elif epoch == "RWD2":
            X_RWD = circmean(X[..., gv.bins_RWD2], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_RWD

    X_epochs = np.moveaxis(X_epochs, 0, -1)
    # print("X_epochs", X_epochs.shape, epochs)

    if X_epochs.shape[-1] == 1:
        X_epochs = X_epochs[..., 0]

    return X_epochs
