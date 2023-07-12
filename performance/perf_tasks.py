#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from common import constants as gv
from common.options import set_options
from common.plot_utils import save_fig, pkl_save

from data.get_data import get_X_y_days

from stats.bootstrap import my_boots_ci
# from statistics.shuffle import my_shuffle


def perf_tasks_days(
    y_days, perf_type="correct_hit", IF_TASKS=1, IF_LASER=0, IF_SAMPLE="all"
):
    perf_task = []
    ci_task = []

    # perf_type = perf_type.lower()

    for i_task in ["DPA", "DualGo", "DualNoGo"]:
        perf_day = []
        ci_day = []

        try:
            idx_paired = ((y_days.sample_odor == 0) & (y_days.test_odor == 0)) | (
                (y_days.sample_odor == 1) & (y_days.test_odor == 1)
            )

            idx_unpaired = ((y_days.sample_odor == 0) & (y_days.test_odor == 1)) | (
                (y_days.sample_odor == 1) & (y_days.test_odor == 0)
            )
        except:
            pass

        if IF_TASKS:
            idx = (y_days.tasks == i_task) & (y_days.laser == IF_LASER)
        else:
            idx = y_days.laser == IF_LASER

        if IF_SAMPLE == "A":
            idx &= y_days.sample_odor == 0
        elif IF_SAMPLE == "B":
            idx &= y_days.sample_odor == 1

        if ("hit" in perf_type) or ("miss" in perf_type):
            idx &= idx_paired
        elif ("rej" in perf_type) or ("fa" in perf_type):
            idx &= idx_unpaired

        y_task = y_days[idx]

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


if __name__ == "__main__":

    options = set_options()
    _, y_days = get_X_y_days()

    perf_type = "correct"
    sample = "B"

    perf_off, ci_off = perf_tasks_days(
        y_days, perf_type=perf_type, IF_TASKS=1, IF_LASER=0, IF_SAMPLE=sample
    )
    # perf_on, ci_on = perf_tasks_days(y_days, perf_type=perf_type, IF_TASKS=1, IF_LASER=1)

    day_list = gv.days

    figname = perf_type + "_tasks" + "_sample_" + sample
    fig = plt.figure(figname)

    plt.plot(day_list, perf_off[0], "-o", color=gv.pal[0], ms=1.5, label="DPA")
    plt.plot(day_list + 0.1, perf_off[1], "-o", color=gv.pal[1], ms=1.5, label="DualGo")
    plt.plot(
        day_list + 0.2, perf_off[2], "-o", color=gv.pal[2], ms=1.5, label="DualNoGo"
    )
    plt.plot([1, gv.days[-1]], [0.5, 0.5], "--k", alpha=0.5)

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
    plt.ylim([0.25, 1.05])
    plt.xticks(gv.days)
    plt.yticks([0.25, 0.5, 0.75, 1])
    plt.xlim([0.8, gv.days[-1] + 0.2])
    # plt.legend(loc='best', frameon=False)

    pkl_save(fig, figname, path=gv.figdir)
    save_fig(fig, figname, path=gv.figdir)
