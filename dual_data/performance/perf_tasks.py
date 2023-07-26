#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from dual_data.common import constants as gv
from dual_data.common.get_data import get_X_y_days, get_X_y_mice
from dual_data.common.options import set_options
from dual_data.common.plot_utils import pkl_save, save_fig
from dual_data.stats.bootstrap import my_boots_ci


def get_labels_task(y, task, perf_type, SAMPLE="all", IF_LASER=0):
    idx = (y.tasks == task) & (y.laser == IF_LASER)

    # if SAMPLE == "A":
    #     idx &= y.sample_odor == 0
    # elif SAMPLE == "B":
    #     idx &= y.sample_odor == 1

    if ("hit" in perf_type) or ("miss" in perf_type):
        idx_paired = ((y.sample_odor == 0) & (y.test_odor == 0)) | (
            (y.sample_odor == 1) & (y.test_odor == 1)
        )

        idx &= idx_paired
    elif ("rej" in perf_type) or ("fa" in perf_type):
        idx_unpaired = ((y.sample_odor == 0) & (y.test_odor == 1)) | (
            (y.sample_odor == 1) & (y.test_odor == 0)
        )
        idx &= idx_unpaired

    return idx


def perf_tasks_days(y_days, perf_type="correct_hit", SAMPLE="all", IF_LASER=0):
    perf_task = []
    ci_task = []

    n_days = len(y_days.day.unique())
    # print(y_days.head())

    for task in ["DPA", "DualGo", "DualNoGo"]:
        perf_day = []
        ci_day = []

        idx = get_labels_task(y_days, task, perf_type, SAMPLE, IF_LASER)
        y_task = y_days[idx]

        print(task, len(idx), y_days.shape, y_task.shape)

        for i_day in range(1, n_days + 1):
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

    perf_task = np.array(perf_task)
    ci_task = np.array(ci_task)

    return perf_task, ci_task


def run_perf_tasks(**kwargs):
    options = set_options(**kwargs)
    if options["mouse"] == "all":
        _, y_days = get_X_y_mice(IF_RELOAD=options["reload"])
    else:
        _, y_days = get_X_y_days(options["mouse"], IF_RELOAD=options["reload"])

    perf_type = options["perf_type"]
    sample = options["sample"]
    laser = options["laser"]

    perf_off, ci_off = perf_tasks_days(y_days, perf_type, SAMPLE=sample, IF_LASER=laser)
    # perf_on, ci_on = perf_tasks_days(y_days, perf_type, SAMPLE=sample, IF_LASER=1)

    n_days = len(y_days.day.unique())
    day_list = np.arange(1, n_days + 1)

    figname = options["mouse"] + "_behavior_tasks_" + perf_type
    fig = plt.figure(figname)

    plt.plot(day_list, perf_off[0], "-o", color=gv.pal[0], ms=1.5, label="DPA")
    plt.plot(day_list + 0.1, perf_off[1], "-o", color=gv.pal[1], ms=1.5, label="DualGo")
    plt.plot(
        day_list + 0.2, perf_off[2], "-o", color=gv.pal[2], ms=1.5, label="DualNoGo"
    )
    plt.plot([1, day_list[-1]], [0.5, 0.5], "--k", alpha=0.5)

    plt.errorbar(
        day_list, perf_off[0], yerr=ci_off[0].T, fmt="none", alpha=1, color=gv.pal[0]
    )
    plt.errorbar(
        day_list + 0.1,
        perf_off[1],
        yerr=ci_off[1].T,
        fmt="none",
        alpha=1,
        color=gv.pal[1],
    )
    plt.errorbar(
        day_list + 0.2,
        perf_off[2],
        yerr=ci_off[2].T,
        fmt="none",
        alpha=1,
        color=gv.pal[2],
    )

    plt.xlabel("Day")
    plt.ylabel(perf_type)
    plt.ylim([0.25, 1.25])
    plt.xticks(gv.days)
    plt.yticks([0.25, 0.5, 0.75, 1])
    plt.xlim([0.8, day_list[-1] + 0.2])
    # plt.legend(loc='best', frameon=False)

    pkl_save(fig, figname, path=gv.figdir)
    save_fig(fig, figname, path=gv.figdir)


if __name__ == "__main__":
    run_perf_tasks()
