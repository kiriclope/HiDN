import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.common.plot_utils import add_vlines


def plot_lick_rate(licks_counts, bin_edges, n_mice=1):
    # convert count of events to rate (Hz) by dividing by the total time in seconds
    bin_widths = np.diff(bin_edges)
    rates = licks_counts / bin_widths / n_mice / 32

    plt.plot(bin_edges[:-1], rates[0], "r")
    plt.plot(bin_edges[:-1], rates[1], "b")
    plt.plot(bin_edges[:-1], rates[2], "g")
    add_vlines2()
    plt.xlabel("Time (s)")
    plt.ylabel("Lick Rate (Hz)")
    # plt.ylim([0, 20])
    # plt.ylim([0, 5])
    # plt.xlim([0, 10])
    # plt.show()

def plot_licks_hist(licks_all, n_bins="auto", n_mice=1):
    # licks_counts, bin_edges = np.histogram(licks_all, bins=n_bins, density=False)
    # print(licks_counts.shape, bin_edges.shape)

    licks_counts, bin_edges, _ = plt.hist(licks_all, bins=n_bins, density=False)
    plt.clf()
    print('lick_count', licks_counts.shape)
    plot_lick_rate(licks_counts, bin_edges, n_mice)
    return licks_counts, bin_edges


def hstack_nan(X, Y):
    # Get dimensions
    rows_x, _ = X.shape
    rows_y, _ = Y.shape

    # Ensure same number of rows via padding
    if rows_x < rows_y:
        X = np.pad(
            X,
            pad_width=((0, rows_y - rows_x), (0, 0)),
            mode="constant",
            constant_values=np.nan,
        )
    elif rows_x > rows_y:
        Y = np.pad(
            Y,
            pad_width=((0, rows_x - rows_y), (0, 0)),
            mode="constant",
            constant_values=np.nan,
        )

    # Stack arrays
    return np.hstack((X, Y))


def vstack_nan(X, Y):
    # Convert the arrays to float data type for supporting NaN
    # X = X.astype(float)
    # Y = Y.astype(float)

    cols_x = X.shape[0]
    cols_y = Y.shape[0]

    if cols_x < cols_y:
        X = np.pad(
            X,
            pad_width=(0, cols_y - cols_x),
            mode="constant",
            constant_values=-100,
        )
    elif cols_x > cols_y:
        Y = np.pad(
            Y,
            pad_width=(0, cols_x - cols_y),
            mode="constant",
            constant_values=-100,
        )

    return np.vstack((X, Y))


def add_vlines2(t_STIM=[0, 1], t_DIST=[2.5, 3.5], t_CUE=[4.5, 5], t_TEST=[7, 8], t_RWD=[5, 5.5], t_RWD2=[9,10], ax=None):
    # def add_vlines(t_STIM=[0, 1], t_DIST=[4, 5], t_CUE=[7, 7.5], t_TEST=[11, 12], ax=None):
    time_periods = [t_STIM, t_DIST, t_CUE, t_RWD, t_TEST, t_RWD2]

    colors = ["b", "b", "g", "y", "b", "y"]

    if ax is None:
        for period, color in zip(time_periods, colors):
            plt.axvspan(period[0], period[1], alpha=0.1, color=color)
    else:
        for period, color in zip(time_periods, colors):
            ax.axvspan(period[0], period[1], alpha=0.1, color=color)


def get_licks_and_times(data, mouse="ACC_Prl"):
    raw_data = data["SerialData"].T
    licks = data["lickTime"][:, 0]

    if ("Prl" in mouse) | ("DPA" in mouse):
        idx_S1_on = (raw_data[2] == 90) & (raw_data[3] == 12)
        idx_S2_on = (raw_data[2] == 89) & (raw_data[3] == 11)

        idx_S1_off = (raw_data[2] == 90) & (raw_data[3] == 0)
        idx_S2_off = (raw_data[2] == 89) & (raw_data[3] == 0)

        idx_Go_on = (raw_data[2] == 10) & (raw_data[3] == 2)
        idx_NoGo_on = (raw_data[2] == 18) & (raw_data[3] == 10)

        idx_Go_off = (raw_data[2] == 10) & (raw_data[3] == 0)
        idx_NoGo_off = (raw_data[2] == 18) & (raw_data[3] == 0)

        idx_T1_on = (raw_data[2] == 15) & (raw_data[3] == 7)
        idx_T2_on = (raw_data[2] == 12) & (raw_data[3] == 4)

        idx_T1_off = (raw_data[2] == 15) & (raw_data[3] == 0)
        idx_T2_off = (raw_data[2] == 12) & (raw_data[3] == 0)

        idx_hit = (raw_data[2] == 7) & (raw_data[3] == 3)
        idx_miss = (raw_data[2] == 6) & (raw_data[3] == 3)
        idx_cr = (raw_data[2] == 5) & (raw_data[3] == 3)
        idx_fa = (raw_data[2] == 4) & (raw_data[3] == 3)
    else:
        idx_S1_on = (raw_data[2] == 16) & (raw_data[3] == 8)
        idx_S2_on = (raw_data[2] == 17) & (raw_data[3] == 9)

        idx_S1_off = (raw_data[2] == 16) & (raw_data[3] == 0)
        idx_S2_off = (raw_data[2] == 17) & (raw_data[3] == 0)

        idx_Go_on = (raw_data[2] == 13) & (raw_data[3] == 5)
        idx_NoGo_on = (raw_data[2] == 14) & (raw_data[3] == 6)

        idx_Go_off = (raw_data[2] == 13) & (raw_data[3] == 0)
        idx_NoGo_off = (raw_data[2] == 14) & (raw_data[3] == 0)

        idx_T1_on = (raw_data[2] == 11) & (raw_data[3] == 3)
        idx_T2_on = (raw_data[2] == 12) & (raw_data[3] == 4)

        idx_T1_off = (raw_data[2] == 11) & (raw_data[3] == 0)
        idx_T2_off = (raw_data[2] == 12) & (raw_data[3] == 0)

        idx_hit = (raw_data[2] == 7) & (raw_data[3] == 3)
        idx_miss = (raw_data[2] == 6) & (raw_data[3] == 3)
        idx_cr = (raw_data[2] == 5) & (raw_data[3] == 3)
        idx_fa = (raw_data[2] == 4) & (raw_data[3] == 3)

    events = raw_data[0]

    t_S1_off = events[idx_S1_off]
    t_S2_off = events[idx_S2_off]

    t_S1_on = events[idx_S1_on]
    t_S2_on = events[idx_S2_on]

    t_Go_on = events[idx_Go_on]
    t_NoGo_on = events[idx_NoGo_on]

    t_Go_off = events[idx_Go_off]
    t_NoGo_off = events[idx_NoGo_off]

    t_T1_on = events[idx_T1_on]
    t_T2_on = events[idx_T2_on]

    t_T1_off = events[idx_T1_off]
    t_T2_off = events[idx_T2_off]

    t_hit = events[idx_hit]
    t_miss = events[idx_miss]
    t_cr = events[idx_cr]
    t_fa = events[idx_fa]

    t_correct = np.sort(np.hstack((t_hit, t_cr)))
    t_incorrect = np.sort(np.hstack((t_miss, t_fa)))

    # t_correct = t_cr
    # t_incorrect = t_fa

    # t_response = np.sort(np.hstack((t_hit, t_miss, t_cr, t_fa)))

    # print("sample")
    t_sample_on = np.sort(np.hstack((t_S1_on, t_S2_on)))
    t_sample_off = np.sort(np.hstack((t_S1_off, t_S2_off)))

    t_sample = vstack_nan(t_sample_on, t_sample_off)
    # print(t_sample.shape)

    # print("dist")
    t_dist_on = np.sort(np.hstack((t_Go_on, t_NoGo_on)))
    t_dist_off = np.sort(np.hstack((t_Go_off, t_NoGo_off)))

    t_dist = vstack_nan(t_dist_on, t_dist_off)

    # print("test")
    t_test_on = np.sort(np.hstack((t_T1_on, t_T2_on)))
    t_test_off = np.sort(np.hstack((t_T1_off, t_T2_off)))

    # print("test", np.array(t_test_on).shape, np.array(t_test_off).shape)
    t_test = vstack_nan(t_test_on, t_test_off)

    # print("go", np.array(t_Go_on).shape, np.array(t_Go_off).shape)
    t_go = vstack_nan(t_Go_on, t_Go_off)

    # print("nogo", np.array(t_NoGo_on).shape, np.array(t_NoGo_off).shape)
    t_nogo = vstack_nan(t_NoGo_on, t_NoGo_off)

    # print(t_sample.shape)

    licks = (licks - t_sample[0][0]) / 1000

    t_dist = (t_dist - t_sample[0][0]) / 1000
    t_test = (t_test - t_sample[0][0]) / 1000
    t_go = (t_go - t_sample[0][0]) / 1000
    t_nogo = (t_nogo - t_sample[0][0]) / 1000

    t_correct = (t_correct - t_sample[0][0]) / 1000
    t_incorrect = (t_incorrect - t_sample[0][0]) / 1000

    t_sample = (t_sample - t_sample[0][0]) / 1000

    return licks, t_sample, t_dist, t_test, t_go, t_nogo, t_correct, t_incorrect


def convert_to_serie(sample_series, distractor_series, test_series):
    sample_index = 0
    distractor_index = 0
    test_index = 0

    time_series = []

    # Iterate while there's data in at least one series
    while (
        (sample_index < len(sample_series))
        or (distractor_index < len(distractor_series))
        or (test_index < len(test_series))
    ):
        sample_time = (
            sample_series[sample_index]
            if sample_index < len(sample_series)
            else float("inf")
        )
        distractor_time = (
            distractor_series[distractor_index]
            if distractor_index < len(distractor_series)
            else float("inf")
        )
        test_time = (
            test_series[test_index] if test_index < len(test_series) else float("inf")
        )

        # find which event happened first, add event as a tuple (event_type, time)
        if sample_time <= distractor_time and sample_time <= test_time:
            time_series.append(("sample", sample_time))
            sample_index += 1
        elif distractor_time <= sample_time and distractor_time <= test_time:
            time_series.append(("distractor", distractor_time))
            distractor_index += 1
        else:  # test_time is smallest
            time_series.append(("test", test_time))
            test_index += 1

    return time_series


def split_trials(series, go_distractors, no_go_distractors, responses, trial_length=21):
    all_trials = []
    trials_go = []
    trials_no_go = []
    trials_without_distractor = []
    labels = []

    i = 0
    while i < len(series):
        if series[i][0] == "sample":
            end_of_trial = (
                series[i][1] + trial_length
            )  # Added trial length to the sample timestamp

            # DPA trial
            if i + 1 < len(series) and series[i + 1][0] == "test":
                if any(
                    end_of_trial > r > series[i + 1][1] for r in responses
                ):  # Check if a response occurred within this trial
                    trials_without_distractor.append((series[i][1], end_of_trial))
                    all_trials.append((series[i][1], end_of_trial))
                    labels.append(0)

                i += 2
            # GNG trials
            elif (
                i + 2 < len(series)
                and series[i + 1][0] == "distractor"
                and series[i + 2][0] == "test"
            ):
                if any(
                    end_of_trial > r > series[i + 2][1] for r in responses
                ):  # Check if a response occurred within this trial
                    all_trials.append((series[i][1], end_of_trial))
                    if series[i + 1][1] in go_distractors:
                        trials_go.append((series[i][1], end_of_trial))
                        labels.append(1)
                    elif series[i + 1][1] in no_go_distractors:
                        trials_no_go.append((series[i][1], end_of_trial))
                        labels.append(2)
                i += 3
            else:
                i += 1
        else:
            i += 1

    return trials_without_distractor, trials_go, trials_no_go, all_trials, labels


def get_licks_in_trial(start_time, end_time, lick_timestamps):
    return [
        time - start_time for time in lick_timestamps if start_time <= time <= end_time
    ]


def hstack_with_padding(arrays):
    try:
        max_length = max((x.shape[0] for x in arrays if not np.all(np.isnan(x))), default=0)
    except:
        max_length = 0

    # pad arrays and stack
    padded_arrays = []
    for a in arrays:
        pad_len = max_length - a.shape[0]
        padded_a = np.pad(
            a, ((0, pad_len),) + ((0, 0),) * (a.ndim - 1), constant_values=np.nan
        )
        padded_arrays.append(padded_a)
    stacked_array = np.hstack(padded_arrays)
    return stacked_array


def pad_lists(list_of_lists):
    max_len = max((len(lst) for lst in list_of_lists if not np.all(np.isnan(lst))), default=0)
    return [lst + [np.nan] * (max_len - len(lst)) for lst in list_of_lists]  # pad lists


def pad_list(sub_list, max_len, pad_value=float("nan")):
    return sub_list + [pad_value] * (max_len - len(sub_list))


def get_licks_array(licks, serie, trial_length):
    licks_trial = [get_licks_in_trial(start, start + trial_length, licks) for start, _ in serie]

    max_len = max((len(x) for x in licks_trial), default=0)
    licks_array = np.array([pad_list(x, max_len) for x in licks_trial])

    return licks_array


def get_licks_mouse(data, mouse, response="", trial_length=21, verbose=1):
    # if verbose:
    #     print("get licks time")

    (
        t_licks,
        t_sample,
        t_dist,
        t_test,
        t_go,
        t_nogo,
        t_correct,
        t_incorrect,
    ) = get_licks_and_times(data, mouse)

    # if verbose:
    #     print("get serie")

    events_serie = convert_to_serie(t_sample[0], t_dist[0], t_test[0])

    # if verbose:
    #     print("get splits")

    if response == "correct":
        t_response = t_correct
    elif response == "incorrect":
        t_response = t_incorrect
    else:
        t_response = np.hstack((t_correct, t_incorrect))

    dpa_trials, go_trials, nogo_trials, all_trials, labels = split_trials(
        events_serie, t_go[0], t_nogo[0], t_response, trial_length
    )

    # if verbose:
    #     print("get licks")

    licks_all = get_licks_array(t_licks, all_trials, trial_length)
    licks_dpa = get_licks_array(t_licks, dpa_trials, trial_length)
    licks_go = get_licks_array(t_licks, go_trials, trial_length)
    licks_nogo = get_licks_array(t_licks, nogo_trials, trial_length)

    if verbose:
        print(
            "licks: all",
            licks_all.shape,
            "licks: DPA",
            licks_dpa.shape,
            "Go",
            licks_go.shape,
            "NoGo",
            licks_nogo.shape,
        )

    return licks_dpa, licks_go, licks_nogo, licks_all, labels


def get_licks_mice(path, n_session=10, response="", trial_length=20, ini=0):
    mice = np.sort(os.listdir(path))
    mice = [mouse for mouse in mice if "DPA" not in mouse]

    mice_dpa = []
    mice_go = []
    mice_nogo = []

    for mouse in mice:
        licks_dpa = []
        licks_go = []
        licks_nogo = []

        print("mouse", mouse)
        #
        for i_session in range(ini, n_session + 1):
            # print(path + mouse + "/session_%d" % i_session)

            data = loadmat(path + mouse + "/session_%d" % i_session)

            dpa, go, nogo = get_licks_mouse(
                data, path, response, trial_length, verbose=0
            )

            licks_dpa.append(dpa)
            licks_go.append(go)
            licks_nogo.append(nogo)

        licks_dpa = hstack_with_padding(licks_dpa)
        licks_go = hstack_with_padding(licks_go)
        licks_nogo = hstack_with_padding(licks_nogo)

        # print("dpa", licks_dpa.shape, "go", licks_go.shape, "nogo", licks_nogo.shape)

        mice_dpa.append(licks_dpa)
        mice_go.append(licks_go)
        mice_nogo.append(licks_nogo)

    return mice_dpa, mice_go, mice_nogo


def get_perf_mice(path, n_session):
    df_mice = pd.DataFrame(
        columns=["hit", "miss", "fa", "cr", "perf", "animal", "session", "task", "opto"]
    )
    mice = np.sort(os.listdir(path))

    for mouse in mice:
        for session in range(n_session + 1):
            try:
                raw = loadmat(path + mouse + "/session_%d" % session)
            except:
                pass

            if "opto" in mouse:
                session = session + 0.2

            if 0 == 1:
                # if 'DPA'in mouse:
                pass
            else:
                # for task in ["Single_DPA", "DPA", "DualGo", "DualNoGo", "ODR"]:
                # for task in ["ODR"]:
                for task in ["ODR", "DPA", "DualGo", "DualNoGo"]:
                    try:
                        if task == "Single_DPA":
                            if "DPA" in mouse:
                                data = np.mean(raw["Data"], 0)[1:]
                            else:
                                data = np.mean(raw["DataP"], 0)[1:]
                        elif task == "DPA":
                            if "DPA" in mouse:
                                pass
                            else:
                                data = np.mean(raw["Data"], 0)[1:]

                        elif task == "DualGo":
                            data = np.mean(raw["DataD1"], 0)[1:]
                        elif task == "DualNoGo":
                            data = np.mean(raw["DataD2"], 0)[1:]
                        elif task == "ODR":
                            data = np.mean(raw["Data1"], 0)[1:]

                        data[-1] /= 100
                        data[:-1] /= np.sum(data[:-1])
                        data[:-1] *= 2

                        data = data[np.newaxis]
                        df = pd.DataFrame(
                            data, columns=["hit", "miss", "fa", "cr", "perf"]
                        )

                        df["animal"] = mouse[-1]
                        df["session"] = session

                        # if task=='DPA':
                        #     df['session'] = session
                        # if task=='DualGo':
                        #     df['session'] = session + .2
                        # if task=='DualNoGo':
                        #     df['session'] = session - .2

                        df["task"] = task

                        if "opto" in mouse:
                            # if path == "../data/behavior/DualTask-Silencing-ACC/":
                            #     df["opto"] = "opto_ACC"
                            # else:
                            df["opto"] = "opto"
                        else:
                            df["opto"] = "control"

                        df_mice = pd.concat([df_mice, df], ignore_index=True)

                    except:
                        pass

    return df_mice
