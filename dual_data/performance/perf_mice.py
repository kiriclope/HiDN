#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

from common import constants as gv
from common.plot_utils import save_fig, pkl_save
from common.get_data import get_X_y_mice, get_X_y_days

from stats.bootstrap import my_boots_ci


def get_labels(y_days, perf_type, task, IF_SAMPLE="all", IF_LASER=0):
    idx = (y_days.tasks == task) & (y_days.laser == IF_LASER)

    # if IF_SAMPLE == "A":
    #     idx &= y_days.sample_odor == 0
    # elif IF_SAMPLE == "B":
    #     idx &= y_days.sample_odor == 1

    if ("hit" in perf_type) or ("miss" in perf_type):
        idx_paired = ((y_days.sample_odor == 0) & (y_days.test_odor == 0)) | (
            (y_days.sample_odor == 1) & (y_days.test_odor == 1)
        )
        idx &= idx_paired
    elif ("rej" in perf_type) or ("fa" in perf_type):
        idx_unpaired = ((y_days.sample_odor == 0) & (y_days.test_odor == 1)) | (
            (y_days.sample_odor == 1) & (y_days.test_odor == 0)
        )
        idx &= idx_unpaired

    y_task = y_days[idx]

    return y_task


def get_perf_tasks_days(y_days, perf_type="correct_hit", IF_LASER=0, IF_SAMPLE="all"):
    perf_task = []
    ci_task = []

    for i_task in ["DPA", "DualGo", "DualNoGo"]:
        perf_day = []
        ci_day = []

        y_task = get_labels(y_days, perf_type, i_task, IF_SAMPLE, IF_LASER)

        for i_day in range(1, gv.n_days + 1):
            y_day = y_task[y_task.day == i_day]

            if perf_type == "correct":
                idx = ~y_day.response.str.contains("incorrect")
            else:
                idx = y_day.response.str.contains(perf_type)

            perf_day.append(np.mean(idx))

            _, ci = my_boots_ci(idx, np.mean, verbose=0, n_samples=1000)
            ci_day.append(ci[0])

        perf_task.append(perf_day)
        ci_task.append(ci_day)

    perf_mice = np.array(perf_task)
    ci_mice = np.array(ci_task)

    return perf_mice, ci_mice


def plot_perf_days(perf_mice, ci_mice, perf_type):
    day_list = gv.days

    figname = perf_type + "_tasks" + "_mice"

    fig = plt.figure(figname)

    labels = ["DPA", "DualGo", "DualNoGo"]
    dx = [0.0, 0.1, 0.2]
    pal = ["r", "b", "g"]

    for i in range(3):
        plt.plot(
            day_list + dx[i],
            perf_mice[i],
            "-o",
            color=pal[i],
            ms=3,
            label=labels[i],
        )

        plt.errorbar(
            day_list + dx[i],
            perf_mice[i],
            yerr=ci_mice[i].T,
            fmt="none",
            alpha=1,
            color=pal[i],
        )

    plt.plot([1, gv.days[-1]], [0.5, 0.5], "--k", alpha=1)
    plt.xticks(gv.days)
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.xlim([0.75, gv.days[-1] + 0.25])

    plt.xlabel("Day")
    if "fa" in perf_type:
        plt.ylabel("False Alarm Rate")
    elif "hit" in perf_type:
        plt.ylabel("Hit Rate")
    elif "miss" in perf_type:
        plt.ylabel("Miss Rate")
    elif "rej" in perf_type:
        plt.ylabel("Correct Rejection Rate")
    else:
        plt.ylabel("Performance")
        plt.ylim([0.45, 1.05])

    plt.legend(loc="best", frameon=False, fontsize=8)

    pkl_save(fig, figname, path=gv.figdir)
    save_fig(fig, figname, path=gv.figdir)


# def get_perf_laser():


if __name__ == "__main__":
    options["features"] = sys.argv[1]
    options["day"] = sys.argv[2]
    options["task"] = sys.argv[3]

    print(gv.mice)
    _, y_days = get_X_y_mice()

    # _, y_days = get_X_y_days()

    perf_type = "correct"
    IF_SAMPLE = "all"

    perf_type = "hit"

    perf_mice, ci_mice = get_perf_tasks_days(
        y_days, perf_type=perf_type, IF_LASER=0, IF_SAMPLE=IF_SAMPLE
    )

    perf_type = "hit_fa"

    plot_perf_days(perf_mice, ci_mice, perf_type)

    perf_type = "fa"

    perf_mice, ci_mice = get_perf_tasks_days(
        y_days, perf_type=perf_type, IF_LASER=0, IF_SAMPLE=IF_SAMPLE
    )

    perf_type = "hit_fa"

    plot_perf_days(perf_mice, ci_mice, perf_type)
