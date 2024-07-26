import multiprocessing
import warnings
import seaborn as sns

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np


def set_options(**kwargs):
    opts = dict()

    opts["path"] = "/home/leon/dual_task/dual_data/"
    opts["fname"] =''

    opts["mouse"] = "JawsM15"
    opts["data_type"] = "raw"  # "raw" or "dF"

    opts["n_days"] = 6  # PrL,ACC 6 or multi 10 this is updated later

    opts["reload"] = False
    opts["n_jobs"] = int(0.90 * multiprocessing.cpu_count())
    opts["verbose"] = 0

    ################################
    # behavior param
    ################################
    opts["perf_type"] = "hit"
    opts["sample"] = "all"

    ################################
    # task param
    ################################
    opts["tasks"] = np.array(["DPA", "DualGo", "DualNoGo"])
    opts["task"] = "DualGo"  # DPA, DualGo, DualNoGo, Dual, or all
    opts["day"] = "first"  # int or 'first', 'middle', 'last'
    opts["trials"] = ""  # 'correct', 'incorrect'
    opts["features"] = "sample"
    opts["num_features"] = 1
    # 'sample', 'distractor', 'task', 'reward', "choice"
    opts["overlap"] = "sample"
    opts["show_AB"] = False
    opts["laser"] = 0

    opts["epochs"] = ["ED", "MD", "LD"]
    opts["AVG_EPOCHS"] = 1

    ################################
    # perm/boots param
    ################################
    opts["bootstrap"] = 0
    opts["shuffle"] = 0

    opts["n_samples"] = 1000  # for permutation test
    opts["n_shuffles"] = 1000  # for permutation test
    opts["n_boots"] = 1000  # for bootstrap
    opts["n_repeats"] = 10  # for repeated Kfold

    opts["n_splits"] = 3
    opts["avg_coefs"] = True

    opts["add_vlines"] = 0

    ################################
    # preprocessing parameters
    ################################
    opts["preprocess"] = 1
    opts["DCVL"] = 0

    # scaling fluo
    opts["scaler_BL"] = "robust"  # standard, robust, center
    opts["center_BL"] = None
    opts["scale_BL"] = None
    opts["avg_mean_BL"] = False
    opts["avg_noise_BL"] = True
    opts["unit_var_BL"] = False

    ################################
    # Synthetic data augmentation
    ################################
    # create synthetic traces with spawner
    opts["augment"] = False
    opts["sig_aug"] = 0.005
    opts["n_aug"] = 1

    # adjust imbalance in trial types
    opts["balance"] = 'under' # under, over, aug
    opts["class_weight"] = 0 # to use pos_weight in nn.BCEwithLogitsLoss and balance the classes

    ################################
    # classification parameters
    ################################
    opts["clf"] = "log_loss"  # "log_loss" or "LinearSVC" or "LDA" or "SGD"
    opts["method"] = None  # None or 'bolasso' or 'bootstrap' or 'gridsearch'

    # precision
    opts["tol"] = 1e-3
    opts["max_iter"] = 5000

    # intercept
    opts["fit_intercept"] = True  # always set to true
    opts["intercept_scaling"] = 1  # unstable if too large

    # penalty
    opts["penalty"] = "l1"  # "l1", "l2" or elasticnet
    opts["alpha"] = 0.5  # between 0 and 1
    opts["bolasso_penalty"] = "l2"
    opts["solver"] = "liblinear"  # liblinear or saga
    opts["class_weight"] = None  # "balanced"  # 'balanced' or None
    opts["refit"] = True  # default true
    opts["multi_class"] = "auto"  # 'auto' or 'multinomial'

    # shrinkage for LDA
    opts["shrinkage"] = "auto"

    # standardization
    opts["scaler"] = None  # 'standard', 'robust', 'center', None
    opts["unit_var"] = False

    # params for SGD
    opts["learning_rate"] = "optimal"  # optimal, adaptative
    opts["l1_ratio"] = 0.15

    ################################
    # Dimensionality reduction
    ################################
    # prescreening
    opts["prescreen"] = 'fpr'  # fpr, fdr, fwe or None
    opts["pval"] = 0.05
    opts["bolasso_pval"] = 0.05

    # PCA
    opts["pca"] = False
    opts["n_comp"] = None  # None or mle or int

    # corr
    opts["corr"] = False
    opts["threshold"] = 0.25

    ################################
    # fit
    ################################
    # outer cv for score estimates
    opts["random_state"] = np.random.randint(1e4)
    opts["out_fold"] = "repeated"  # stratified, loo, repeated
    opts["n_out"] = 5
    opts["outer_score"] = "f1_weighted"
    opts["scoring"] = 'roc_auc'
    # accuracy, roc_auc, f1_macro, f1_weighted

    # inner cv for hyperparam tuning
    opts["in_fold"] = "stratified"  # stratified, loo, repeated
    opts["n_in"] = 5
    opts["inner_score"] = "f1_weighted"
    # accuracy, roc_auc, f1_macro, f1_weighted, neg_log_loss

    # multiclass/label
    opts["multilabel"] = False
    opts["multiclass"] = False

    # gridsearch params
    opts["n_lambda"] = 50
    opts["n_alpha"] = 10

    inv_frame = 0  # 1 / frame_rate
    opts['T_WINDOW'] = 0.5

    opts.update(kwargs)

    opts["Cs"] = np.logspace(-3, 3, opts["n_lambda"])
    opts["alphas"] = np.linspace(0, 1, opts["n_alpha"])

    opts["data_path"] = opts["path"] + "data/"
    opts["fig_path"] = opts["path"] + "figs/"

    opts["n_days"] = 6  # PrL,ACC 6 or multi 10 this is updated later
    opts["n_discard"] = 0
    opts["n_first"] = 3  # 3 or 2
    opts["n_middle"] = 0  # 0 or 2

    if "P" in opts["mouse"]:
        opts["n_days"] = 10  # PrL 6, ACC 5 or multi 10
        opts["n_discard"] = 1
        opts["n_first"] = 3  # 3 or 2
        opts["n_middle"] = 3 # 0 or 2
    if "ACC" in opts["mouse"]:
        opts["n_days"] = 5  # PrL 6, ACC 5 or multi 10
    if "17" in opts["mouse"]:
        opts["n_days"] = 8  # PrL 6, ACC 5 or multi 10
        opts["n_discard"] = 0
        opts["n_first"] = 3  # 3 or 2
        opts["n_middle"] = 3 # 0 or 2

    if opts["day"] == "first":
        palette = sns.color_palette("muted")
    else:
        palette = sns.color_palette("bright")

    frame_rate = 6

    opts['duration'] = 14  # 14, 19.2
    if "P" in opts["mouse"]:
        opts['duration'] = 14

    time = np.linspace(0, opts['duration'], int( opts['duration'] * frame_rate))
    opts["bins"] = np.arange(0, len(time))

    t_BL = [0, 2]
    t_STIM = [2 + inv_frame, 3]
    t_ED = [3 + inv_frame, 4.5]
    t_DIST = [4.5 + inv_frame, 5.5]
    t_MD = [5.5 + inv_frame, 6.5]
    t_CUE = [6.5 + inv_frame, 7]
    t_RWD = [7 + inv_frame, 7.5]
    t_LD = [7.5 + inv_frame, 9]
    t_TEST = [9 + inv_frame, 10]
    t_RWD2 = [11 + inv_frame, 12]

    if "P" in opts["mouse"]:
        t_BL = [0 + inv_frame, 2]
        t_STIM = [2 + inv_frame, 3]
        t_ED = [3 + inv_frame, 4]
        t_DIST = [4 + inv_frame, 5]
        t_MD = [5 + inv_frame, 6]
        t_CUE = [6 + inv_frame, 7]
        t_RWD = [7 + inv_frame, 8]
        t_LD = [8 + inv_frame, 9]
        t_TEST = [9 + inv_frame, 10]
        t_RWD2 = [11 + inv_frame, 12]

    opts["bins_BL"] = opts["bins"][int(t_BL[0] * frame_rate) : int(t_BL[1] * frame_rate)]

    opts["bins_STIM"]= opts["bins"][int((t_STIM[0] + opts['T_WINDOW']) * frame_rate) : int(t_STIM[1] * frame_rate)]

    opts["bins_ED"] = opts["bins"][int((t_ED[0] + opts['T_WINDOW']) * frame_rate) : int((t_ED[1]) * frame_rate)]

    opts["bins_STIM_ED"] = opts["bins"][int((t_STIM[0] + opts['T_WINDOW']) * frame_rate) : int((t_ED[1]) * frame_rate)]

    opts["bins_DELAY"] = opts["bins"][int((t_ED[0] + opts['T_WINDOW']) * frame_rate) : int((t_TEST[0]) * frame_rate)]

    opts["bins_PRE_DIST"] = opts["bins"][int((t_BL[0] + opts['T_WINDOW']) * frame_rate) : int((t_DIST[0]) * frame_rate)]

    opts["bins_POST_DIST"] = opts["bins"][int((t_DIST[0] + opts['T_WINDOW']) * frame_rate) : int((t_TEST[0]) * frame_rate)]

    opts["bins_DIST"] = opts["bins"][int((t_DIST[0] + opts['T_WINDOW']) * frame_rate) : int(t_DIST[1] * frame_rate)]

    opts["bins_MD"]= opts["bins"][int((t_MD[0] + opts['T_WINDOW']) * frame_rate) : int((t_MD[1]) * frame_rate)]

    opts["bins_CUE"] = opts["bins"][int((t_CUE[0] + opts['T_WINDOW']) * frame_rate) : int(t_CUE[1] * frame_rate)]

    opts["bins_RWD"] = opts["bins"][int((t_RWD[0]) * frame_rate) : int(t_RWD[1] * frame_rate)]

    opts["bins_LD"] = opts["bins"][int((t_LD[0] + opts['T_WINDOW']) * frame_rate) : int(t_LD[1] * frame_rate)]

    opts["bins_TEST"] = opts["bins"][
        int((t_TEST[0] + opts['T_WINDOW']) * frame_rate) : int((t_TEST[1]) * frame_rate)
    ]

    opts["bins_CHOICE"] = opts["bins"][
        int((t_TEST[1]) * frame_rate) : int((t_RWD2[1] - opts['T_WINDOW']) * frame_rate)
    ]

    opts["bins_RWD2"] = opts["bins"][int((t_RWD2[0] + opts['T_WINDOW']) * frame_rate) : int(t_RWD2[1] * frame_rate)]

    opts["pal"] = [
        palette[3],
        palette[0],
        palette[2],
        palette[1],
    ]

    opts["paldict"] = {
        "DPA": palette[3],
        "DualGo": palette[0],
        "DualNoGo": palette[2],
        "Dual": palette[1],
        "all": palette[4],
    }

    return opts
