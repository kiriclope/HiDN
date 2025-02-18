import glob
import os
import pickle
from importlib import reload

import mat73
import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.common import constants as gv
from src.preprocess.helpers import preprocess_df

reload(gv)

def find_mat_files(folder_path):
    mat_files = np.array(glob.glob(os.path.join(folder_path, '*.mat')))
    return mat_files[0]

def get_gng_behavior(mouse, day):
    path = '/home/leon/dual_task/dual_data/data/2Samples-DualTask-BehavioralData'
    cols = ['sample', 'test', 'response', 'pair', 'tasks', 'cue', 'odr_response', 'odr_pair', 'laser']
    file_name = path + '/%s-DualTask-BehavioralData' % mouse + '/day_%d/' % day
    file = find_mat_files(file_name)

    data = loadmat(file)
    df = pd.DataFrame(data['AllTrials']['AllTrials'][0][0], columns=cols)

    df.loc[df.odr_response==1, 'odr_choice'] = 1
    df.loc[df.odr_response==2, 'odr_choice'] = 0
    df.loc[df.odr_response==3, 'odr_choice'] = 1
    df.loc[df.odr_response==4, 'odr_choice'] = 0

    df.loc[df.odr_response==1, 'odr_perf'] = 1
    df.loc[df.odr_response==2, 'odr_perf'] = 0
    df.loc[df.odr_response==3, 'odr_perf'] = 0
    df.loc[df.odr_response==4, 'odr_perf'] = 1

    return df[['odr_response', 'odr_choice', 'odr_perf']]

def get_X_y_day_new(idx_day, data, data_type="raw"):

    if "raw" in data_type:
        X_days = np.rollaxis(data["Cdf_Trial"], 1, 0)
    elif "FR" in data_type:
        X_days = np.rollaxis(data["FR_Trial"], 1, 0)
    else:
        try:
            X_days = np.rollaxis(data["dff_TrialBase"], 1, 0)
        except:
            X_days = np.rollaxis(data["dFF_Trial"], 1, 0)

    y_ = np.zeros((X_days.shape[0], 7))
    y_days = pd.DataFrame(
        y_, columns=["sample_odor", "dist_odor", "test_odor", "tasks", "response", "laser", "day"]
    )

    y_days.loc[data["S1Trial"].flatten() - 1, 'sample_odor'] = 0
    y_days.loc[data["S2Trial"].flatten() - 1, 'sample_odor'] = 1

    try:
        y_days.loc[data["S3Trial"].flatten() - 1, 'sample_odor'] = 2
        y_days.loc[data["S4Trial"].flatten() - 1, 'sample_odor'] = 3
    except:
        pass

    try:
        y_days.loc[data["NDTrial"].flatten() - 1, 'dist_odor'] = np.nan
        y_days.loc[data["NDTrial"].flatten() - 1, 'tasks'] = "DPA"
    except:
        y_days.loc[data["ODPATrial"].flatten() - 1, 'dist_odor'] = np.nan
        y_days.loc[data["ODPATrial"].flatten() - 1, 'tasks'] = "DPA"

    y_days.loc[data["D1Trial"].flatten() - 1, 'dist_odor'] = 0
    y_days.loc[data["D2Trial"].flatten() - 1, 'dist_odor'] = 1

    y_days.loc[data["D1Trial"].flatten() - 1, 'tasks'] = "DualGo"
    y_days.loc[data["D2Trial"].flatten() - 1, 'tasks'] = "DualNoGo"

    try:
        y_days.loc[data["hitTrial"].flatten() - 1, 'response'] = "correct_hit"
        y_days.loc[data["CRTrial"].flatten() - 1, 'response'] = "correct_rej"
        y_days.loc[data["FATrial"].flatten() - 1, 'response'] = "incorrect_fa"
        y_days.loc[data["missTrial"].flatten() - 1, 'response'] = "incorrect_miss"

        y_days.loc[data["hitTrial"].flatten() - 1, 'choice'] = 1
        y_days.loc[data["CRTrial"].flatten() - 1, 'choice'] = 0
        y_days.loc[data["FATrial"].flatten() - 1, 'choice'] = 1
        y_days.loc[data["missTrial"].flatten() - 1, 'choice'] = 0

        y_days.loc[data["hitTrial"].flatten() - 1, 'pair'] = 1
        y_days.loc[data["CRTrial"].flatten() - 1, 'pair'] = 0
        y_days.loc[data["FATrial"].flatten() - 1, 'pair'] = 0
        y_days.loc[data["missTrial"].flatten() - 1, 'pair'] = 1

    except:

        y_days.loc[data["ODPAHitTrial"].flatten() - 1, 'response'] = "correct_hit"
        y_days.loc[data["ODPACRTrial"].flatten() - 1, 'response'] = "correct_rej"
        y_days.loc[data["ODPAFATrial"].flatten() - 1, 'response'] = "incorrect_fa"

        y_days.loc[data["ODPAHitTrial"].flatten() - 1, 'choice'] = 1
        y_days.loc[data["ODPACRTrial"].flatten() - 1, 'choice'] = 0
        y_days.loc[data["ODPAFATrial"].flatten() - 1, 'choice'] = 1

        y_days.loc[data["ODPAHitTrial"].flatten() - 1, 'pair'] = 1
        y_days.loc[data["ODPACRTrial"].flatten() - 1, 'pair'] = 0
        y_days.loc[data["ODPAFATrial"].flatten() - 1, 'pair'] = 0

        y_days.loc[data["ODRHitTrial"].flatten() - 1, 'odr_perf'] = 1
        y_days.loc[data["ODRCRTrial"].flatten() - 1, 'odr_perf'] = 1

        y_days.loc[data["ODRHitTrial"].flatten() - 1, 'odr_choice'] = 1
        y_days.loc[data["ODRCRTrial"].flatten() - 1, 'odr_choice'] = 0

        y_days.loc[data["ODRHitTrial"].flatten() - 1, 'odr_response'] = 1
        y_days.loc[data["ODRCRTrial"].flatten() - 1, 'odr_response'] = 4

        try:
            y_days.loc[data["ODRFATrial"].flatten() - 1, 'odr_response'] = 3
            y_days.loc[data["ODRFATrial"].flatten() - 1, 'odr_perf'] = 0
            y_days.loc[data["ODRFATrial"].flatten() - 1, 'odr_choice'] = 1
        except:
            pass

        try:
            y_days.loc[data["ODPAMissTrial"].flatten() - 1, 'response'] = "incorrect_miss"
            y_days.loc[data["ODPAMissTrial"].flatten() - 1, 'choice'] = 0
            y_days.loc[data["ODPAMissTrial"].flatten() - 1, 'pair'] = 1
        except:
            pass

        try:
            y_days.loc[data["ODRMissTrial"].flatten() - 1, 'odr_perf'] = 0
            y_days.loc[data["ODRMissTrial"].flatten() - 1, 'odr_response'] = 2
            y_days.loc[data["ODRMissTrial"].flatten() - 1, 'odr_choice'] = 0
        except:
            pass

    y_days.loc[(y_days.response=="correct_hit") & (y_days.sample_odor==0), 'test_odor'] = 0
    y_days.loc[(y_days.response=="incorrect_fa") & (y_days.sample_odor==0), 'test_odor'] = 1
    y_days.loc[(y_days.response=="correct_rej") & (y_days.sample_odor==0), 'test_odor'] = 1
    y_days.loc[(y_days.response=="incorrect_miss") & (y_days.sample_odor==0), 'test_odor'] = 0

    y_days.loc[(y_days.response=="correct_hit") & (y_days.sample_odor==1), 'test_odor'] = 1
    y_days.loc[(y_days.response=="incorrect_fa") & (y_days.sample_odor==1), 'test_odor'] = 0
    y_days.loc[(y_days.response=="correct_rej") & (y_days.sample_odor==1), 'test_odor'] = 0
    y_days.loc[(y_days.response=="incorrect_miss") & (y_days.sample_odor==1), 'test_odor'] = 1

    y_days.loc[(y_days.response=="correct_hit") & (y_days.sample_odor==2), 'test_odor'] = 0
    y_days.loc[(y_days.response=="incorrect_fa") & (y_days.sample_odor==2), 'test_odor'] = 1
    y_days.loc[(y_days.response=="correct_rej") & (y_days.sample_odor==2), 'test_odor'] = 1
    y_days.loc[(y_days.response=="incorrect_miss") & (y_days.sample_odor==2), 'test_odor'] = 0

    y_days.loc[(y_days.response=="correct_hit") & (y_days.sample_odor==3), 'test_odor'] = 1
    y_days.loc[(y_days.response=="incorrect_fa") & (y_days.sample_odor==3), 'test_odor'] = 0
    y_days.loc[(y_days.response=="correct_rej") & (y_days.sample_odor==3), 'test_odor'] = 0
    y_days.loc[(y_days.response=="incorrect_miss") & (y_days.sample_odor==3), 'test_odor'] = 1

    try:
        y_days.loc[data["laserOffTrial"].flatten() - 1, 'laser'] = 0
        y_days.loc[data["laserOnTrial"].flatten() - 1, 'laser'] = 1
    except:
        pass

    y_days['day'] = idx_day

    return X_days, y_days

def get_X_y_days_multi(mouse=gv.mouse):

    data = mat73.loadmat(
        "/home/leon/dual_task/dual_data/data/%s/dataProcessed.mat" % mouse
    )

    X_days = np.swapaxes(data["Cdf_Mice"], 0, 1)
    y_ = np.zeros((X_days.shape[0], 7))
    y_days = pd.DataFrame(
        y_, columns=["sample_odor", "dist_odor", "test_odor", "tasks", "response", "laser", "day"]
    )

    y_days.sample_odor[data["S1All"][0] - 1] = 0
    y_days.sample_odor[data["S2All"][0] - 1] = 1
    y_days.sample_odor[data["S3All"][0] - 1] = 2
    y_days.sample_odor[data["S4All"][0] - 1] = 3

    y_days.dist_odor[data["NDAll"][0] - 1] = np.nan
    y_days.dist_odor[data["D1All"][0] - 1] = 0
    y_days.dist_odor[data["D2All"][0] - 1] = 1
    y_days.dist_odor[data["D3All"][0] - 1] = 2
    y_days.dist_odor[data["D4All"][0] - 1] = 3

    y_days.tasks[data["NDAll"][0] - 1] = "DPA"
    y_days.tasks[data["D1All"][0] - 1] = "DualGo"
    y_days.tasks[data["D2All"][0] - 1] = "DualGo"
    y_days.tasks[data["D3All"][0] - 1] = "DualNoGo"
    y_days.tasks[data["D4All"][0] - 1] = "DualNoGo"

    y_days.response[data["AllCorrect"][0] - 1] = "correct"
    y_days.response[data["AllWrong"][0] - 1] = "incorrect"

    y_days.test_odor[(y_days.response=="correct") & (y_days.sample_odor==0)] = 0
    y_days.test_odor[(y_days.response=="incorrect") & (y_days.sample_odor==0)] = 1

    y_days.test_odor[(y_days.response=="correct") & (y_days.sample_odor==1)] = 1
    y_days.test_odor[(y_days.response=="incorrect") & (y_days.sample_odor==1)] = 0

    y_days.test_odor[(y_days.response=="correct") & (y_days.sample_odor==2)] = 0
    y_days.test_odor[(y_days.response=="incorrect") & (y_days.sample_odor==2)] = 1

    y_days.test_odor[(y_days.response=="correct") & (y_days.sample_odor==3)] = 1
    y_days.test_odor[(y_days.response=="incorrect") & (y_days.sample_odor==3)] = 0

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

    # print(y_.shape)

    if day is None:
        y_df = pd.DataFrame(
            y_.T, columns=["sample_odor", "dist_odor", "test_odor", "response", "tasks", "laser"]
        )
    else:
        y_ = np.vstack((y_, day * np.ones(y_.shape[-1])))
        y_df = pd.DataFrame(
            y_.T,
            columns=["sample_odor", "test_odor", "response", "tasks", "laser", "day"],
        )

    y_df.loc[y_df.sample_odor == 17, "sample_odor"] = 0
    y_df.loc[y_df.sample_odor == 18, "sample_odor"] = 1

    y_df.loc[y_df.test_odor == 11, "test_odor"] = 0
    y_df.loc[y_df.test_odor == 12, "test_odor"] = 1

    y_df.loc[y_df.tasks == 0, "dist_odor"] = np.nan
    y_df.loc[y_df.tasks == 13, "dist_odor"] = 0
    y_df.loc[y_df.tasks == 14, "dist_odor"] = 1

    y_df.loc[y_df.response == 1, "choice"] = 1
    y_df.loc[y_df.response == 2, "choice"] = 0
    y_df.loc[y_df.response == 3, "choice"] = 1
    y_df.loc[y_df.response == 4, "choice"] = 0

    y_df.loc[y_df.response == 1, "response"] = "correct_hit"
    y_df.loc[y_df.response == 2, "response"] = "incorrect_miss"
    y_df.loc[y_df.response == 3, "response"] = "incorrect_fa"
    y_df.loc[y_df.response == 4, "response"] = "correct_rej"

    y_df.loc[y_df.tasks == 0, "tasks"] = "DPA"
    y_df.loc[y_df.tasks == 13, "tasks"] = "DualGo"
    y_df.loc[y_df.tasks == 14, "tasks"] = "DualNoGo"

    y_df['performance'] = y_df['response'].apply(lambda x: 0 if 'incorrect' in x else 1)
    y_df['pair'] = y_df['response'].apply(lambda x: 0 if (('rej' in x) or ('fa' in x)) else 1)

    return y_df


def get_fluo_data(idx_day, **kwargs):
    """returns X_raw, y_raw from fluorescence data"""

    mouse = kwargs["mouse"]
    path = kwargs["data_path"]
    data_type = kwargs["data_type"]
    n_days = kwargs["n_days"]

    if "ACC" in mouse:
        data = loadmat(path + "/" + mouse + "/SamedROI/" + mouse + "_all_days" + ".mat")

    elif kwargs['NEW_DATA']:

        file = path + "/" + mouse + "/SamedROI_%.2dDays/" % n_days + "Day%.2d" % idx_day + "/DFF_Data01.mat"
        print(file)
        data = mat73.loadmat(file)

    else:
        data = loadmat(
            path
            + "/"
            + mouse
            + "/SamedROI_0%dDays/" % n_days
            + mouse
            + "_day_"
            + str(idx_day)
            + ".mat"
        )

    if kwargs['NEW_DATA']:
        X_raw, y_raw = get_X_y_day_new(idx_day, data, data_type)
    else:
        if "raw" in data_type:
            X_raw = np.rollaxis(data["Cdf_Mice"], 1, 0)
        else:
            X_raw = np.rollaxis(data["dff_Mice"], 1, 0)

        y_raw = data["Events"].transpose()

        if "ACC" in mouse:
            # print(X_raw.shape)
            X_raw = X_raw.reshape(
                (n_days, int(X_raw.shape[0] / n_days), X_raw.shape[1], X_raw.shape[2])
            )
            # print(X_raw.shape)
            y_raw = y_raw.T.reshape(
                (n_days, int(y_raw.T.shape[0] / n_days), y_raw.T.shape[1])
            )

            X_raw = X_raw[idx_day - 1]
            y_raw = y_raw[idx_day - 1].T

    print(
        "mouse",
        mouse,
        "n_days",
        n_days,
        "day",
        idx_day,
        "type",
        data_type,
        "all data: X",
        X_raw.shape,
        "y",
        y_raw.shape,
    )

    return X_raw, y_raw

def get_X_y_days(**kwargs):
    path = kwargs["data_path"]
    mouse = kwargs["mouse"]

    # if "P" in kwargs["mouse"]:
    #     kwargs["n_days"] = 10  # PrL 6, ACC 5 or multi 10
    # if "ACC" in kwargs["mouse"]:
    #     kwargs["n_days"] = 5  # PrL 6, ACC 5 or multi 10
    # if "23" in kwargs["mouse"]:
    #     kwargs["n_days"] = 5  # PrL 6, ACC 5 or multi 10
    # if "17" in kwargs["mouse"]:
    #     kwargs["n_days"] = 8  # PrL 6, ACC 5 or multi 10


    n_days = kwargs["n_days"]
    days = np.arange(1, n_days + 1)
    # print(days)

    if kwargs["reload"] == 0:
        try:
            if kwargs['verbose']:
                print("Loading files from", path + mouse)
            X_days = pickle.load(open(path + mouse + "/X_days.pkl", "rb"))
            y_days = pd.read_pickle(path + mouse + "/y_days.pkl")
        except:
            kwargs["reload"] = 1

    if kwargs["reload"] == 1:
        if kwargs['verbose']:
            print("Reading data from source file")

        if ("AP" in mouse):
            X_days, y_days = get_X_y_days_multi(mouse)
        else:
            X_days = []
            y_days = []

            for day in days:
                X, y = get_fluo_data(idx_day=day, **kwargs)
                # print("X", X.shape, "y", y.shape)

                if kwargs["NEW_DATA"]:
                    y_df = y
                else:
                    y_df = create_df(y, day=day)
                    y_be = get_gng_behavior(mouse, day)
                    y_df['odr_response'] = y_be['odr_response']
                    y_df['odr_choice'] = y_be['odr_choice']
                    y_df['odr_perf'] = y_be['odr_perf']

                X_days.append(X)
                y_days.append(y_df)

            X_days = np.vstack(X_days)
            y_days = pd.concat(y_days, axis=0, ignore_index=True)

        pickle.dump(X_days, open(path + mouse + "/X_days.pkl", "wb"))
        y_days.to_pickle(path + mouse + "/y_days.pkl")

        # print("X_days", X_days.shape, "y_days", y_days.shape)

    if kwargs["preprocess"]:
        if kwargs['verbose']:
            print(
                "PREPROCESSING:",
                "SCALER",
                kwargs["scaler_BL"],
                "AVG MEAN",
                kwargs["avg_mean_BL"],
                "AVG NOISE",
                kwargs["avg_noise_BL"],
                "UNIT VAR",
                kwargs["unit_var_BL"],
            )

        X_days = preprocess_df(X_days, y_days, **kwargs)

    return X_days[..., :84], y_days


def get_X_y_mice(**kwargs):
    mice = kwargs["mice"]
    path = kwargs["data_path"]
    IF_RELOAD = kwargs['reload']

    if IF_RELOAD == 0:
        print("Loading files from", path + "mice")
        X_mice = pickle.load(open(path + "mice" + "/X_mice.pkl", "rb"))
        y_mice = pd.read_pickle(path + "mice" + "/y_mice.pkl")
    else:
        X_mice = []
        y_mice = []
        for mouse in mice:
            kwargs['mouse'] = mouse
            X_days, y_days = get_X_y_days(**kwargs)

            X_mice.append(X_days)
            y_days["mouse"] = mouse
            y_mice.append(y_days)

        X_mice = np.vstack(np.array(X_days))
        y_mice = pd.concat(y_mice, axis=0, ignore_index=True)

        pickle.dump(X_mice, open(path + "mice" + "/X_mice.pkl", "wb"))
        y_mice.to_pickle(path + "mice" + "/y_mice.pkl")

    return X_mice, y_mice


def get_X_y_S1_S2(X, y, **kwargs):
    if kwargs['verbose']:
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

    idx_trials = True
    idx_S3 = False
    idx_S4 = False

    if kwargs["trials"] == "correct":
        idx_trials = ~y.response.str.contains("incorrect")
    elif kwargs["trials"] == "incorrect":
        idx_trials = y.response.str.contains("incorrect")

    idx_laser = True
    if kwargs["laser"] == 1:
        idx_laser = y.laser == 1
    elif kwargs["laser"] == 0:
        idx_laser = y.laser == 0
    elif kwargs["laser"] == -1:
        idx_laser = True

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

        # if kwargs['trials'] == 'incorrect':
        #     idx = y.response.str.contains("incorrect")
        #     idx_S1 = (y.sample_odor == 0) & idx & (y.laser == 0)
        #     idx_S2 = (y.sample_odor == 1) & idx & (y.laser == 0)
        #     # idx_S3 = y.laser == 1
        #     idx_trials = True
        #     # idx_laser = True

        if kwargs["multilabel"]:
            idx_S3 = y.sample_odor == 2
            idx_S4 = y.sample_odor == 3

    elif kwargs["features"] == "pair":
        idx_S1 = y.pair == 1
        idx_S2 = y.pair == 0

    elif kwargs["features"] == "fa":
        # lick
        idx_S1 = y.response == "correct_rej"
        # no lick
        idx_S2 = y.response == "incorrect_fa"

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

    elif kwargs["features"] == "choice":
        idx_S1 = y.choice == 1
        idx_S2 = y.choice == 0

    elif kwargs["features"] == "odr_choice":
        idx_S1 = y.odr_choice == 1
        idx_S2 = y.odr_choice == 0

        idx_trials = True
        if kwargs["trials"] == "correct":
            idx_trials = y.odr_perf == 1
        elif kwargs["trials"] == "incorrect":
            idx_tasks = True
            idx_S1 = y.odr_perf == 0
            idx_S2 = y.tasks == "DPA"


    elif kwargs["features"] == "reward":
        idx_S1 = ~y.response.str.contains("incorrect")
        idx_S2 = y.response.str.contains("incorrect")

    elif kwargs["features"] == "test":
        idx_S1 = y.test_odor == 0
        idx_S2 = y.test_odor == 1

    elif kwargs["features"] == "distractor":
        idx_S1 = y.dist_odor == 0
        idx_S2 = y.dist_odor == 1

        idx_trials = True
        if kwargs["trials"] == "correct":
            idx_trials = y.odr_perf == 1
        elif kwargs["trials"] == "incorrect":
            idx_tasks = True
            idx_S1 = y.odr_perf == 0
            idx_S2 = y.tasks == "DPA"

        if kwargs["multilabel"]:
            idx_S3 = y.dist_odor == 2
            idx_S4 = y.dist_odor == 3

    elif kwargs["features"] == "task":
        idx_S1 = y.tasks == "DPA"
        if kwargs["task"] == "Dual":
            idx_S2 = (y.tasks == "DualGo") | (y.tasks == "DualNoGo")
        else:
            idx_S2 = y.tasks == kwargs["task"]

    if isinstance(kwargs["day"], str):
        print("multiple days, discard", kwargs["n_discard"],
              'first', kwargs["n_first"], 'middle', kwargs["n_middle"])
        if kwargs["day"] == "first":
            idx_days = (y.day > kwargs["n_discard"]) & (y.day <= kwargs["n_first"] + kwargs["n_discard"])

        if kwargs["day"] == "middle":
            idx_days = (y.day > kwargs["n_first"] + kwargs["n_discard"]) & (
                y.day <= kwargs["n_first"] + kwargs["n_middle"] + kwargs["n_discard"]
            )
        if kwargs["day"] == "last":
            idx_days = y.day > (kwargs["n_first"] + kwargs["n_middle"] + kwargs["n_discard"])
    else:
        idx_days = y.day == kwargs["day"]

    X_S1 = X[idx_S1 & idx_trials & idx_days & idx_laser & idx_tasks]
    X_S2 = X[idx_S2 & idx_trials & idx_days & idx_laser & idx_tasks]

    X_S3 = X[idx_S3 & idx_trials & idx_days & idx_laser & idx_tasks]
    X_S4 = X[idx_S4 & idx_trials & idx_days & idx_laser & idx_tasks]

    print("X_S1", X_S1.shape, "X_S2", X_S2.shape)
    if X_S3.shape[0] > 0:
        print("X_S3", X_S3.shape, "X_S4", X_S4.shape)

    X_S1_S2 = np.vstack((X_S1, X_S2, X_S3, X_S4))

    y_S1 = y[idx_S1 & idx_trials & idx_days & idx_laser & idx_tasks]
    y_S2 = y[idx_S2 & idx_trials & idx_days & idx_laser & idx_tasks]
    y_S3 = y[idx_S3 & idx_trials & idx_days & idx_laser & idx_tasks]
    y_S4 = y[idx_S4 & idx_trials & idx_days & idx_laser & idx_tasks]

    y_S1_S2 = pd.concat((y_S1, y_S2, y_S3, y_S4))

    return X_S1_S2 * 1000.0, y_S1_S2
