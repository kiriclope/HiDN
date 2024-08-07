import matplotlib.pyplot as plt
import numpy as np

from src.common import constants as gv
from src.common.get_data import get_X_y_days, get_X_y_mice
from src.common.options import set_options
from src.common.plot_utils import pkl_save, save_fig
from src.stats.bootstrap import my_boots_ci


def get_labels_task(y, task, perf_type, SAMPLE="all", IF_LASER=0):
    idx = (y.tasks == task) & (y.laser == IF_LASER)

    if SAMPLE == "A":
        idx &= y.sample_odor == 0
    elif SAMPLE == "B":
        idx &= y.sample_odor == 1

    if SAMPLE == "C":
        idx &= y.test_odor == 0
    elif SAMPLE == "D":
        idx &= y.test_odor == 1

    if ("hit" in perf_type) or ("miss" in perf_type):
        idx_paired = ((y.sample_odor == 0) & (y.test_odor == 0)) | (
            (y.sample_odor == 1) & (y.test_odor == 1)
        )

        print(np.mean(idx_paired))

        idx &= idx_paired
    elif ("rej" in perf_type) or ("fa" in perf_type):
        idx_unpaired = ((y.sample_odor == 0) & (y.test_odor == 1)) | (
            (y.sample_odor == 1) & (y.test_odor == 0)
        )
        idx &= idx_unpaired

        print(np.mean(idx_unpaired))

    return idx


def perf_tasks_days(tasks, y_days, perf_type="correct_hit", SAMPLE="all", IF_LASER=0):
    perf_task = []
    ci_task = []

    n_days = len(y_days.day.unique())
    # print(y_days.head())

    for task in tasks:
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

            # perf_day.append(np.sum(idx))
            # _, ci = my_boots_ci(idx, np.sum, verbose=0, n_samples=1000)

            ci_day.append(ci[0])
            # ci_day.append(0)

        perf_task.append(perf_day)
        ci_task.append(ci_day)

    perf_task = np.array(perf_task)
    ci_task = np.array(ci_task)

    return perf_task, ci_task

def df_perf_tasks_days(tasks, y_days, perf_type="correct_hit", SAMPLE="all", IF_LASER=0):
    perf_task = []
    ci_task = []

    n_days = len(y_days.day.unique())
    # print(y_days.head())

    for task in tasks:
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

            # perf_day.append(np.sum(idx))
            # _, ci = my_boots_ci(idx, np.sum, verbose=0, n_samples=1000)

            ci_day.append(ci[0])
            # ci_day.append(0)

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
        # _, y_days = get_X_y_days(options["mouse"], IF_RELOAD=options["reload"])
        _, y_days = get_X_y_days(**options)

    perf_type = options["perf_type"]
    sample = options["sample"]
    laser = options["laser"]
    tasks = options["tasks"]
    perf_off, ci_off = perf_tasks_days(tasks, y_days, perf_type, SAMPLE=sample, IF_LASER=laser)
    # perf_on, ci_on = perf_tasks_days(y_days, perf_type, SAMPLE=sample, IF_LASER=1)

    n_days = len(y_days.day.unique())
    day_list = np.arange(1, n_days + 1)

    figname = options["mouse"] + "_behavior_tasks_" + perf_type
    fig = plt.figure(figname)

    for i in range(len(tasks)):
        if tasks[i] == "DPA":
            plt.plot(day_list, perf_off[i], "-o", color=kwargs['pal'][i], ms=5, label="DPA")
            plt.errorbar(day_list, perf_off[i], yerr=ci_off[i].T, fmt="none", alpha=1, color=kwargs['pal'][0])
        elif tasks[i] == "DualGo":
            plt.plot(day_list + 0.1, perf_off[i], "-o", color=kwargs['pal'][1], ms=5, label="DualGo")
            plt.errorbar(
                day_list + 0.1,
                perf_off[i],
                yerr=ci_off[i].T,
                fmt="none",
                alpha=1,
                color=kwargs['pal'][1],
            )
        elif tasks[i] == "DualNoGo":
            plt.plot(
                day_list + 0.2, perf_off[i], "-o", color=kwargs['pal'][2], ms=5, label="DualNoGo"
            )
            plt.errorbar(
                day_list + 0.2,
                perf_off[i],
                yerr=ci_off[i].T,
                fmt="none",
                alpha=1,
                color=kwargs['pal'][2],
            )
        else:
            plt.plot(day_list, perf_off[i], "-o", color='k', ms=5, label="all")
            plt.errorbar(
                day_list, perf_off[i], yerr=ci_off[i].T, fmt="none", alpha=1, color=kwargs['pal'][i]
            )

    plt.plot([1, day_list[-1]], [0.5, 0.5], "--k", alpha=0.5)

    plt.xlabel("Day")
    plt.ylabel('Correct Rejections')
    plt.ylim([0.25, 1.25])
    plt.xticks(gv.days)

    if 'correct' in perf_type:
        plt.ylabel('Performance')
        if 'hit' in perf_type:
            plt.ylabel('Correct Hit')
        if "rej" in perf_type:
            plt.ylabel('Correct Rejection')
    else:
        if 'miss' in perf_type:
            plt.ylabel('Miss')
        if "fa" in perf_type:
            plt.ylabel('False Alarm')

    if "rej" in perf_type:
        plt.yticks([0., 0.25, 0.5, 0.75, 1])
    else:
        plt.yticks([0.4, 0.6, 0.8, 1])
        plt.ylim([0.4, 1])

    plt.xlim([0.8, options['n_days'] + 0.25])
    # plt.legend(loc='best', frameon=False)

    pkl_save(fig, figname, path=gv.figdir)
    save_fig(fig, figname, path=gv.figdir)

    return perf_off

if __name__ == "__main__":
    run_perf_tasks()
