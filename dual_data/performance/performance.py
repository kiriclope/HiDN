#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from dual_data.common import constants as gv
from dual_data.common.get_data import get_X_y_days
from dual_data.common.options import set_options
from dual_data.stats.bootstrap import my_boots_ci
from dual_data.stats.shuffle import my_shuffle


def perf_tasks_days(y_days, perf_type="correct_hit", IF_TASKS=1, IF_LASER=0):
    perf_task = []
    ci_task = []

    # perf_type = perf_type.lower()

    for i_task in ["DPA", "DualGo", "DualNoGo"]:
        perf_day = []
        ci_day = []

        idx_paired = ((y_days.sample_odor == 0) & (y_days.test_odor == 0)) | (
            (y_days.sample_odor == 1) & (y_days.test_odor == 1)
        )

        idx_unpaired = ((y_days.sample_odor == 0) & (y_days.test_odor == 1)) | (
            (y_days.sample_odor == 1) & (y_days.test_odor == 0)
        )

        if IF_TASKS:
            idx = (y_days.tasks == i_task) & (y_days.laser == IF_LASER)
        else:
            idx = y_days.laser == IF_LASER

        if ("hit" in perf_type) or ("miss" in perf_type):
            idx &= idx_paired
        elif ("rej" in perf_type) or ("fa" in perf_type):
            idx &= idx_unpaired

        y_task = y_days[idx]

        for i_day in range(1, 7):
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
    print(perf_mice.shape, ci_mice.shape)

    return perf_mice, ci_mice


if __name__ == "__main__":
    options = set_options()
    _, y_days = get_X_y_days(IF_RELOAD=1)

    perf_type = "correct"

    perf_off, ci_off = perf_tasks_days(
        y_days, perf_type=perf_type, IF_TASKS=1, IF_LASER=0
    )
    perf_on, ci_on = perf_tasks_days(
        y_days, perf_type=perf_type, IF_TASKS=1, IF_LASER=1
    )

    day_list = np.array([1, 2, 3, 4, 5, 6])

    figname = perf_type + "_off_on"
    fig = plt.figure(figname)

    plt.plot(day_list, perf_off[1], "-o", color=gv.pal[1], ms=1.5)
    plt.plot(day_list + 0.1, perf_on[1], "-o", color=gv.pal[2], ms=1.5)

    plt.plot([0, 6], [0.5, 0.5], "--k", alpha=0.5)

    plt.errorbar(
        day_list, perf_off[1], yerr=ci_off[1].T, fmt="none", alpha=1, color=gv.pal[1]
    )
    plt.errorbar(
        day_list + 0.1,
        perf_on[1],
        yerr=ci_on[1].T,
        fmt="none",
        alpha=1,
        color=gv.pal[2],
    )

    plt.xlabel("Day")
    plt.ylabel(perf_type)
    plt.ylim([0.25, 1.05])
    plt.xticks([1, 2, 3, 4, 5, 6])
    plt.yticks([0.25, 0.5, 0.75, 1])
    plt.xlim([0.8, 6.2])

    plt.savefig(gv.figdir + figname + ".svg", dpi=300, format="svg")
