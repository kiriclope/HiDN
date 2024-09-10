import numpy as np
from scipy import stats
from scipy.stats import circmean, norm
from sklearn.preprocessing import MinMaxScaler
# from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
from oasis.functions import deconvolve
from src.common import constants as gv
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt

# from scipy.signal import savgol_filter, detrend

def minmax_X_y(X, y):
    print("X", X.shape, "y", y.shape)
    X1 = X[y == 1]
    X0 = X[y == 0]

    for i in range(X.shape[2]):
        X1[..., i] = MinMaxScaler().fit_transform(X1[..., i])
        X0[..., i] = MinMaxScaler().fit_transform(X0[..., i]) * -1

    return np.vstack((X0, X1))


def center_BL(X, center=None, avg_mean=0):
    if center is None:
        X_BL = X[..., gv.bins_BL]

        if avg_mean:
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
            center = np.mean(np.hstack(X_BL), axis=-1)
        else:
            center = np.nanmean(X_BL, axis=-1)

    if scale is None:
        if avg_noise:
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
            center = np.nanmedian(np.hstack(X_BL), axis=-1)
        else:
            center = np.nanmedian(X_BL, axis=-1)

    if scale is None:
        if avg_noise:
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
    y=None,
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
    elif scaler == "minmax":
        X_scale = minmax_X_y(X, y)
    else:
        X_scale = X

    # X_scale = savgol_filter(X_scale, int(np.ceil(gv.frame_rate/2.0) * 2 + 1), polyorder = gv.SAVGOL_ORDER, deriv=0, axis=-1, mode='mirror')

    # X_scale = detrend(X_scale, bp=[gv.bins_STIM[0], gv.bins_DIST[0]])

    if return_center_scale:
        return X_scale[: X_S1.shape[0]], X_scale[X_S1.shape[0] :], center, scale
    else:
        return X_scale[: X_S1.shape[0]], X_scale[X_S1.shape[0] :]

def low_pass(signal):

    # Define the sampling rate and cutoff frequency
    sampling_rate = 6  # Hz
    cutoff_freq = 0.01  # Hz

    # Normalize the cutoff frequency with respect to Nyquist frequency
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist

    # Create a Butterworth low-pass filter
    b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)

    # Apply the filter along the last axis
    return filtfilt(b, a, signal, axis=-1)


def preprocess_X(
    X,
    y=None,
    scaler="standard",
    center=None,
    scale=None,
    avg_mean=0,
    avg_noise=0,
    unit_var=0,
):
    # X = savgol_filter(X, int(np.ceil(gv.frame_rate/2.0) * 2 + 1), polyorder = gv.SAVGOL_ORDER, deriv=0, axis=-1, mode='mirror')
    if scaler=="mean":
        X_scale = X - X.mean((0,-1))[np.newaxis, ..., np.newaxis]
    elif scaler=="lowpass":
        X_scale = low_pass(X)
    elif scaler == "standard":
        X_scale, center, scale = standard_scaler_BL(
            X, center, scale, avg_mean, avg_noise
        )
    elif scaler == "robust":
        X_scale = robust_scaler_BL(X, center, scale, avg_mean, avg_noise, unit_var)
    elif scaler == "center":
        X_scale = center_BL(X, center, avg_mean)
    elif scaler == "dff":
        X_scale, center = df_f_scaler_BL(X, center, avg_mean)
    elif scaler == "minmax":
        X_scale = minmax_X_y(X, y)
    else:
        X_scale = X

    # X_scale = savgol_filter(X_scale, int(np.ceil(gv.frame_rate/2.0) * 2 + 1), polyorder = gv.SAVGOL_ORDER, deriv=0, axis=-1, mode='mirror')

    # X_scale = detrend(X_scale, bp=[gv.bins_STIM[0], gv.bins_DIST[0]])

    # print("##########################################")
    # print(
    #     "PREPROCESSING:",
    #     "SCALER",
    #     scaler,
    #     "AVG MEAN",
    #     bool(avg_mean),
    #     "AVG NOISE",
    #     bool(avg_noise),
    #     "UNIT VAR",
    #     bool(unit_var),
    # )
    # print("##########################################")

    return X_scale

def deconvolve_trial_neuron(X):
    BL = np.mean(X[:11])
    c, s, b, g, lam = deconvolve(X, g=(None,None), penalty=1)
    return s

def dcvl_df(X):
    print('Deconvolve Fluo')
    n_trials, n_neurons, n_time = X.shape

    # Initialize an array to hold the deconvolved data
    results = Parallel(n_jobs=-1)(
        delayed(deconvolve_trial_neuron)(X[trial, neuron])
        for trial in range(n_trials) for neuron in range(n_neurons)
    )

    spike_estimates = np.array(results).reshape(n_trials, n_neurons, n_time)

    return spike_estimates


def preprocess_df(X, y, **kwargs):
    n_days = kwargs["n_days"]
    days = np.arange(1, n_days + 1)

    if kwargs['DCVL']:
        X = dcvl_df(X)

    X_scaled = np.zeros(X.shape)
    for day in days:
        idx = y.day == day
        X_day = X[idx]
        X_scaled[idx] = preprocess_X(
            X_day,
            scaler=kwargs["scaler_BL"],
            avg_mean=kwargs["avg_mean_BL"],
            avg_noise=kwargs["avg_noise_BL"],
            unit_var=kwargs["unit_var_BL"],
        )

    return X_scaled


def avg_epochs(X, axis=-1, **kwargs):
    X_avg = np.nanmean(X, axis=axis)
    # print(X_avg.shape)

    X_epochs = np.empty(tuple([len(kwargs["epochs"])]) + X_avg.shape)
    # print(X_epochs.shape)
    for i_epoch, epoch in enumerate(kwargs["epochs"]):
        if epoch == "BL":
            X_BL = np.nanmean(X[..., kwargs["bins_BL"]], axis=axis)
            X_epochs[i_epoch] = X_BL
        elif epoch == "STIM":
            X_STIM = np.nanmean(X[..., kwargs["bins_STIM"]], axis=axis)
            X_epochs[i_epoch] = X_STIM
        elif epoch == "STIM_ED":
            X_STIM = np.nanmean(X[..., kwargs["bins_STIM_ED"]], axis=axis)
            X_epochs[i_epoch] = X_STIM
        elif epoch == "ED":
            X_ED = np.nanmean(X[..., kwargs["bins_ED"]], axis=axis)
            X_epochs[i_epoch] = X_ED
        elif epoch == "STIM_ED":
            X_STIM = np.nanmean(X[..., kwargs["bins_STIM_ED"]], axis=axis)
            X_epochs[i_epoch] = X_STIM
        elif epoch == "DIST":
            X_DIST = np.nanmean(X[..., kwargs["bins_DIST"]], axis=axis)
            X_epochs[i_epoch] = X_DIST
        elif epoch == "PRE_DIST":
            X_DIST = np.nanmean(X[..., kwargs["bins_PRE_DIST"]], axis=axis)
            X_epochs[i_epoch] = X_DIST
        elif epoch == "POST_DIST":
            X_DIST = np.nanmean(X[..., kwargs["bins_POST_DIST"]], axis=axis)
            X_epochs[i_epoch] = X_DIST
        elif epoch == "MD":
            X_MD = np.nanmean(X[..., kwargs["bins_MD"]], axis=axis)
            X_epochs[i_epoch] = X_MD
        elif epoch == "CUE":
            X_CUE = np.nanmean(X[..., kwargs["bins_CUE"]], axis=axis)
            X_epochs[i_epoch] = X_CUE
        elif epoch == "RWD":
            X_RWD = np.nanmean(X[..., kwargs["bins_RWD"]], axis=axis)
            X_epochs[i_epoch] = X_RWD
        elif epoch == "LD":
            X_LD = np.nanmean(X[..., kwargs["bins_LD"]], axis=axis)
            X_epochs[i_epoch] = X_LD
        elif epoch == "TEST":
            X_TEST = np.nanmean(X[..., kwargs["bins_TEST"]], axis=axis)
            X_epochs[i_epoch] = X_TEST
        elif epoch == "CHOICE":
            X_RWD = np.nanmean(X[..., kwargs["bins_CHOICE"]], axis=axis)
            X_epochs[i_epoch] = X_RWD
        elif epoch == "RWD2":
            X_RWD = np.nanmean(X[..., kwargs["bins_RWD2"]], axis=axis)
            X_epochs[i_epoch] = X_RWD
        elif epoch == "DELAY":
            X_DELAY = np.nanmean(X[..., kwargs["bins_DELAY"]], axis=axis)
            X_epochs[i_epoch] = X_DELAY

    X_epochs = np.moveaxis(X_epochs, 0, axis)

    if X_epochs.shape[-1] == 1:
        X_epochs = X_epochs[..., 0]

    return X_epochs


def avg_phase_epochs(X, **kwargs):
    X_avg = np.nanmean(X, axis=-1)

    X_epochs = np.empty(tuple([len(epochs)]) + X_avg.shape)
    # print('X', X_epochs.shape, 'X_avg', X_avg.shape)
    # print('start', gv.bin_start, 'epochs', gv.epochs)
    # print('average over epochs', epochs)

    for i_epoch, epoch in enumerate(epochs):
        if epoch == "BL":
            X_BL = circmean(X[..., kwargs["bins_BL"]], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_BL
        elif epoch == "STIM":
            X_STIM = circmean(X[..., kwargs["bins_STIM"]], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_STIM
        elif epoch == "STIM_ED":
            X_STIM_ED = circmean(
                X[..., kwargs["bins_STIM"] + kwargs["bins_ED"]], axis=-1, high=360, low=0
            )
            X_epochs[i_epoch] = X_STIM_ED
        elif epoch == "ED":
            X_ED = circmean(X[..., kwargs["bins_ED"]], axis=-1, high=360, low=0)
            # print('X_ED', X_ED.shape, 'bins', kwargs["bins_ED)
            X_epochs[i_epoch] = X_ED
        elif epoch == "DIST":
            X_DIST = circmean(X[..., kwargs["bins_DIST"]], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_DIST
        elif epoch == "MD":
            X_MD = circmean(X[..., kwargs["bins_MD"]], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_MD
        elif epoch == "CUE":
            X_CUE = circmean(X[..., kwargs["bins_CUE"]], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_CUE
        elif epoch == "LD":
            X_LD = circmean(X[..., kwargs["bins_LD"]], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_LD
        elif epoch == "RWD":
            X_RWD = circmean(X[..., kwargs["bins_RWD"]], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_RWD
        elif epoch == "TEST":
            X_TEST = circmean(X[..., kwargs["bins_TEST"]], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_TEST
        elif epoch == "DELAY":
            X_DELAY = circmean(X[..., kwargs["bins_DELAY"]], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_DELAY
        elif epoch == "Before":
            X_bef = circmean(X[..., gv.time < kwargs["t_DIST[0]"]], axis=-1, high=360, low=0)
            print(X_bef.shape)
            X_epochs[i_epoch] = X_bef
        elif epoch == "After":
            X_bef = circmean(
                X[..., np.where(gv.time > kwargs["t_DIST[1]"])], axis=-1, high=360, low=0
            )
            X_epochs[i_epoch] = X_bef
        elif epoch == "RWD2":
            X_RWD = circmean(X[..., kwargs["bins_RWD2"]], axis=-1, high=360, low=0)
            X_epochs[i_epoch] = X_RWD

    X_epochs = np.moveaxis(X_epochs, 0, -1)
    # print("X_epochs", X_epochs.shape, epochs)

    if X_epochs.shape[-1] == 1:
        X_epochs = X_epochs[..., 0]

    return X_epochs
