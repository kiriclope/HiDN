from importlib import reload
import pickle
import numpy as np
import pandas as pd
import mat73
from scipy.io import loadmat

from dual_data.preprocess.helpers import preprocess_X, avg_epochs
from dual_data.common import constants as gv

reload(gv)


def get_X_y_days_multi():
    data = mat73.loadmat(
        "/home/leon/dual_task/dual_data/data/%s/dataProcessed.mat" % gv.mouse
    )

    X_days = np.swapaxes(data["Cdf_Mice"], 0, 1)
    y_ = np.zeros((X_days.shape[0], 6))
    y_days = pd.DataFrame(
        y_, columns=["sample_odor", "dist_odor", "tasks", "response", "laser", "day"]
    )

    y_days.sample_odor[data["S1All"][0] - 1] = 0
    y_days.sample_odor[data["S2All"][0] - 1] = 1
    y_days.sample_odor[data["S3All"][0] - 1] = 2
    y_days.sample_odor[data["S4All"][0] - 1] = 3

    y_days.dist_odor[data["NDAll"][0] - 1] = 0
    y_days.dist_odor[data["D1All"][0] - 1] = 1
    y_days.dist_odor[data["D2All"][0] - 1] = 2
    y_days.dist_odor[data["D3All"][0] - 1] = 3
    y_days.dist_odor[data["D4All"][0] - 1] = 4

    y_days.tasks[data["NDAll"][0] - 1] = "DPA"
    y_days.tasks[data["D1All"][0] - 1] = "DualGo"
    y_days.tasks[data["D2All"][0] - 1] = "DualGo"
    y_days.tasks[data["D3All"][0] - 1] = "DualNoGo"
    y_days.tasks[data["D4All"][0] - 1] = "DualNoGo"

    y_days.response[data["AllCorrect"][0] - 1] = "correct"
    y_days.response[data["AllWrong"][0] - 1] = "incorrect"

    try:
        y_days.laser[data["OffAll"][0] - 1] = 0
        y_days.laser[data["OnAll"][0] - 1] = 1
    except:
        pass

    idx = np.arange(0, 11) * 176 + 88
    idx[0] = 0

    for i in range(10):
        y_days.day[idx[i] : idx[i + 1]] = i + 1

    return X_days, y_days


def create_df(y_raw, day=None):
    y_ = np.delete(y_raw, [3, 5, 6, 7], axis=0)

    print(y_.shape)

    if day is None:
        y_df = pd.DataFrame(
            y_.T, columns=["sample_odor", "test_odor", "response", "tasks", "laser"]
        )
    else:
        y_ = np.vstack((y_, day * np.ones(y_.shape[-1])))
        y_df = pd.DataFrame(
            y_.T,
            columns=["sample_odor", "test_odor", "response", "tasks", "laser", "day"],
        )

    y_df.sample_odor[y_df.sample_odor == 17] = 0
    y_df.sample_odor[y_df.sample_odor == 18] = 1

    y_df.test_odor[y_df.test_odor == 11] = 0
    y_df.test_odor[y_df.test_odor == 12] = 1

    y_df.response[y_df.response == 1] = "correct_hit"
    y_df.response[y_df.response == 2] = "incorrect_miss"
    y_df.response[y_df.response == 3] = "incorrect_fa"
    y_df.response[y_df.response == 4] = "correct_rej"

    y_df.tasks[y_df.tasks == 0] = "DPA"
    y_df.tasks[y_df.tasks == 13] = "DualGo"
    y_df.tasks[y_df.tasks == 14] = "DualNoGo"

    return y_df


def get_fluo_data():
    """returns X_raw, y_raw from fluorescence data"""

    if "ACC" in gv.mouse:
        data = loadmat(
            gv.data_path
            + "/"
            + gv.mouse
            + "/SamedROI/"
            + gv.mouse
            + "_all_days"
            + ".mat"
        )
    else:
        data = loadmat(
            gv.data_path
            + "/"
            + gv.mouse
            + "/SamedROI_0%dDays/" % gv.n_days
            + gv.mouse
            + "_day_"
            + str(gv.day)
            + ".mat"
        )

    if "raw" in gv.data_type:
        print("raw")
        X_raw = np.rollaxis(data["Cdf_Mice"], 1, 0)
    else:
        print("dF")
        X_raw = np.rollaxis(data["dff_Mice"], 1, 0)

    y_raw = data["Events"].transpose()

    if "ACC" in gv.mouse:
        print(
            "mouse",
            gv.mouse,
            "days",
            gv.days,
            "type",
            gv.data_type,
            "all data: X",
            X_raw.shape,
            "y",
            y_raw.shape,
        )

        X_raw = X_raw.reshape(
            (6, int(X_raw.shape[0] / 6), X_raw.shape[1], X_raw.shape[2])
        )
        y_raw = y_raw.T.reshape((6, int(y_raw.T.shape[0] / 6), y_raw.T.shape[1]))

        X_raw = X_raw[gv.day - 1]
        y_raw = y_raw[gv.day - 1].T

        print("X", X_raw.shape, "y", y_raw.shape)

    print(
        "mouse",
        gv.mouse,
        "day",
        gv.day,
        "type",
        gv.data_type,
        "all data: X",
        X_raw.shape,
        "y",
        y_raw.shape,
    )

    return X_raw, y_raw


def get_X_y_days(mouse=gv.mouse, IF_PREP=0, IF_AVG=0, IF_RELOAD=0):
    # print(gv.mouse)

    if IF_RELOAD == 0:
        try:
            print("loading files from", gv.filedir + gv.mouse)
            X_days = pickle.load(open(gv.filedir + gv.mouse + "/X_days.pkl", "rb"))
            y_days = pd.read_pickle(gv.filedir + gv.mouse + "/y_days.pkl")
        except:
            IF_RELOAD = 1

    if IF_RELOAD == 1:
        # if 0 == 0:
        print("reading raw data")

        if ("AP" in gv.mouse) or ("PP" in gv.mouse):
            X_days, y_days = get_X_y_days_multi()
        else:
            X_days = []
            y_days = []

            for gv.day in gv.days:
                X, y = get_fluo_data()
                print(X.shape, y.shape)

                y_df = create_df(y, day=gv.day)

                X_days.append(X)
                y_days.append(y_df)

            X_days = np.vstack(np.array(X_days))
            y_days = pd.concat(y_days, axis=0, ignore_index=True)

        pickle.dump(X_days, open(gv.filedir + gv.mouse + "/X_days.pkl", "wb"))
        y_days.to_pickle(gv.filedir + gv.mouse + "/y_days.pkl")

    # if IF_PREP:
    #     X_days = preprocess_X(
    #         X_days, scaler="robust", avg_mean=0, avg_noise=1, unit_var=1
    #     )
    if IF_AVG:
        X_days = avg_epochs(X_days, epochs=None)

    return X_days, y_days


def get_X_y_mice(IF_RELOAD=0):
    if IF_RELOAD == 0:
        print("loading files from", gv.filedir + "mice")
        X_mice = pickle.load(open(gv.filedir + "mice" + "/X_mice.pkl", "rb"))
        y_mice = pd.read_pickle(gv.filedir + "mice" + "/y_mice.pkl")
    else:
        X_mice = []
        y_mice = []
        for gv.mouse in gv.mice:
            X_days, y_days = get_X_y_days()

            X_mice.append(X_days)
            y_days["mouse"] = gv.mouse
            y_mice.append(y_days)

        X_mice = np.vstack(np.array(X_days))
        y_mice = pd.concat(y_mice, axis=0, ignore_index=True)

        pickle.dump(X_mice, open(gv.filedir + "mice" + "/X_mice.pkl", "wb"))
        y_mice.to_pickle(gv.filedir + "mice" + "/y_mice.pkl")

    return X_mice, y_mice


def get_X_y_S1_S2(X, y, **kwargs):
    print("##########################################")
    print(
        "DATA:",
        "FEATURES",
        kwargs["features"],
        "TASK",
        kwargs["task"],
        "TRIALS",
        kwargs["trials"],
        "DAYS",
        kwargs["day"],
        "LASER",
        kwargs["laser"],
    )
    print("##########################################")

    idx_trials = True
    if kwargs["trials"] == "correct":
        idx_trials = ~y.response.str.contains("incorrect")
    elif kwargs["trials"] == "incorrect":
        idx_trials = y.response.str.contains("incorrect")

    idx_tasks = True
    if kwargs["task"] == "DPA":
        idx_tasks = y.tasks == "DPA"
    if kwargs["task"] == "Dual":
        idx_tasks = (y.tasks == "DualGo") | (y.tasks == "DualNoGo")
    if kwargs["task"] == "DualGo":
        idx_tasks = y.tasks == "DualGo"
    if kwargs["task"] == "DualNoGo":
        idx_tasks = y.tasks == "DualNoGo"

    if kwargs["features"] == "sample":
        idx_S1 = y.sample_odor == 0
        idx_S2 = y.sample_odor == 1
        idx_S3 = False
        idx_S4 = False

        if kwargs["multilabel"]:
            idx_S3 = y.sample_odor == 2
            idx_S4 = y.sample_odor == 3

    elif kwargs["features"] == "paired":
        # pair
        idx_S1 = (y.response == "correct_hit") | (y.response == "incorrect_miss")
        # unpair
        idx_S2 = (y.response == "incorrect_fa") | (y.response == "correct_rej")
        idx_S3 = False
        idx_S4 = False

        idx_trials = True

    elif kwargs["features"] == "choice":
        # lick
        idx_S1 = (y.response == "correct_hit") | (y.response == "incorrect_fa")
        # no lick
        idx_S2 = (y.response == "incorrect_miss") | (y.response == "correct_rej")
        idx_S3 = False
        idx_S4 = False

        idx_trials = True

    elif kwargs["features"] == "fa":
        # lick
        idx_S1 = y.response == "correct_rej"
        # no lick
        idx_S2 = y.response == "incorrect_fa"
        idx_S3 = False
        idx_S4 = False

        idx_trials = True
    elif kwargs["features"] == "decision":
        if kwargs["trials"] == "correct":
            # lick
            idx_S1 = y.response == "correct_hit"
            # no lick
            idx_S2 = y.response == "correct_rej"
        else:
            # lick
            idx_S1 = y.response == "incorrect_fa"
            # no lick
            idx_S2 = y.response == "incorrect_miss"

        idx_S3 = False
        idx_S4 = False

        idx_trials = True

    elif kwargs["features"] == "reward":
        idx_S1 = ~y.response.str.contains("incorrect")
        idx_S2 = y.response.str.contains("incorrect")
        idx_S3 = False
        idx_S4 = False

        idx_trials = True

    elif kwargs["features"] == "test":
        idx_S1 = y.test_odor == 0
        idx_S2 = y.test_odor == 1
        idx_S3 = False
        idx_S4 = False

    elif kwargs["features"] == "distractor":
        idx_S1 = y.tasks == "DualGo"
        idx_S2 = y.tasks == "DualNoGo"
        idx_S3 = False
        idx_S4 = False

        if kwargs["multilabel"]:
            idx_S1 = y.dist_odor == 1
            idx_S2 = y.dist_odor == 2
            idx_S3 = y.dist_odor == 3
            idx_S4 = y.dist_odor == 4

        idx_tasks = True
    elif kwargs["features"] == "task":
        idx_S1 = y.tasks == "DPA"
        if kwargs["task"] == "Dual":
            idx_S2 = (y.tasks == "DualGo") | (y.tasks == "DualNoGo")
        else:
            idx_S2 = y.tasks == kwargs["task"]
        idx_S3 = False
        idx_S4 = False
        idx_tasks = True

    idx_days = True
    if isinstance(kwargs["day"], str):
        print("multiple days")
        if kwargs["day"] == "first":
            idx_days = (y.day > gv.n_discard) & (y.day <= gv.n_first + gv.n_discard)

        if kwargs["day"] == "middle":
            idx_days = (y.day > gv.n_first + gv.n_discard) & (
                y.day <= gv.n_first + gv.n_middle + gv.n_discard
            )
        if kwargs["day"] == "last":
            idx_days = y.day > (gv.n_first + gv.n_middle + gv.n_discard)
            # idx_days = (y.day == 4) | (y.day == 6)
    else:
        print("single day")
        idx_days = y.day == kwargs["day"]

    idx_laser = True

    if kwargs["laser"] == 1:
        idx_laser = y.laser == 1
    elif kwargs["laser"] == 0:
        idx_laser = y.laser == 0

    X_S1 = X[idx_S1 & idx_trials & idx_days & idx_laser & idx_tasks]
    X_S2 = X[idx_S2 & idx_trials & idx_days & idx_laser & idx_tasks]

    X_S3 = X[idx_S3 & idx_trials & idx_days & idx_laser & idx_tasks]
    X_S4 = X[idx_S4 & idx_trials & idx_days & idx_laser & idx_tasks]

    print("X_S1", X_S1.shape, "X_S2", X_S2.shape)
    if X_S3.shape[0] > 0:
        print("X_S3", X_S3.shape, "X_S4", X_S4.shape)

    X_S1_S2 = np.vstack((X_S1, X_S2, X_S3, X_S4))

    if kwargs["multilabel"]:
        # This is the multiclass version of the problem
        # since cross validation doesn t work otherwise
        # y_S1_S2 = np.hstack((np.zeros(kwargs['n_S1_Go']), np.ones(kwargs['n_S1_NoGo']),
        #                      2*np.ones(kwargs['n_S2_Go']), 3*np.ones(kwargs['n_S2_NoGo'])))

        y_S1_S2 = np.hstack(
            (
                np.zeros(X_S1.shape[0]),
                np.ones(X_S2.shape[0]),
                2 * np.ones(X_S3.shape[0]),
                3 * np.ones(X_S4.shape[0]),
            )
        )

    elif kwargs["multiclass"]:
        y_S1_S2 = np.hstack(
            (
                np.zeros(kwargs["n_S1"]),
                np.ones(kwargs["n_S1_Go"]),
                2 * np.ones(kwargs["n_S2"]),
                3 * np.ones(kwargs["n_S2_Go"]),
            )
        )
    else:
        y_S1_S2 = np.hstack((np.zeros(X_S1.shape[0]), np.ones(X_S2.shape[0])))

    return X_S1_S2, y_S1_S2
