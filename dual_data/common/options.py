import multiprocessing

import numpy as np


def set_options(**kwargs):
    opts = dict()

    opts["reload"] = False
    opts["n_jobs"] = int(0.90 * multiprocessing.cpu_count())
    opts["verbose"] = 0

    ################################
    # behavior param
    ################################
    opts['perf_type'] = 'hit'
    opts['sample'] = 'all'

    ################################
    # task param
    ################################
    opts["mouse"] = "JawsM18"
    opts["tasks"] = np.array(["DPA", "DualGo", "DualNoGo"])
    opts["task"] = "DualGo"  # DPA, DualGo, DualNoGo, Dual, or all
    opts["day"] = "first"  # int or 'first', 'middle', 'last'

    opts["trials"] = "correct"  # 'correct', 'incorrect'

    opts["features"] = "sample"  # 'sample', 'distractor', 'task', 'reward', "choice"
    opts["overlap"] = "sample"

    opts["laser"] = 0

    ################################
    # perm/boots param
    ################################
    opts["n_samples"] = 1000  # for permutation test
    opts["n_shuffles"] = 1000  # for permutation test
    opts["n_boots"] = 100  # for bootstrap
    opts["n_repeats"] = 10  # for repeated Kfold

    opts["avg_coefs"] = True
    opts["bootstrap"] = 0
    opts["shuffle"] = 0

    opts["add_vlines"] = 0

    ################################
    # preprocessing parameters
    ################################
    opts["epochs"] = ["ED", "MD", "LD"]
    opts["epoch_sample"] = ["ED"]
    opts["epoch_dist"] = ["MD"]
    opts["epoch_choice"] = ["MD"]
    opts["epoch_rwd"] = ["MD"]

    # scaling fluo
    opts["scaler_BL"] = None  # standard, robust, center
    opts["center_BL"] = None
    opts["scale_BL"] = None
    opts["avg_mean_BL"] = 0
    opts["avg_noise_BL"] = 1
    opts["unit_var_BL"] = 1
    opts["return_center_scale"] = 0

    ################################
    # Synthetic data augmentation
    ################################
    # create synthetic traces with spawner
    opts["augment"] = False
    opts["sig_aug"] = 0.005
    opts["n_aug"] = 1

    # adjust imbalance in trial types
    opts["balance"] = True
    opts["imbalance"] = False

    ################################
    # classification parameters
    ################################
    opts["clf"] = "log_loss"  # "log_loss" or "LinearSVC" or "LDA" or "SGD"
    opts["method"] = None  # None or 'bolasso' or 'bootstrap' or 'gridsearch'

    # precision
    opts["tol"] = 1e-3
    opts["max_iter"] = int(1e4)

    # intercept
    opts["fit_intercept"] = True  # always set to true
    opts["intercept_scaling"] = 1  # unstable if too large

    # penalty
    opts["penalty"] = "l1"  # "l1", "l2" or elasticnet
    opts["solver"] = "liblinear"  # liblinear or saga
    opts["class_weight"] = "balanced"  # "balanced"  # 'balanced' or None
    opts["refit"] = True  # default true
    opts["multi_class"] = "auto"  # 'auto' or 'multinomial'

    opts["n_lambda"] = 10
    opts["alpha"] = 0.5  # between 0 and 1
    opts["n_alpha"] = 50

    # shrinkage for LDA
    opts["shrinkage"] = "auto"

    # standardization
    opts["standardize"] = None  # 'standard', 'robust', 'center', None

    # params for SGD
    opts["learning_rate"] = "optimal"  # optimal, adaptative
    opts["l1_ratio"] = 0.15

    # prescreening
    opts["prescreen"] = False
    opts["pval"] = 0.05

    # PCA
    opts["pca"] = False
    opts["n_comp"] = None  # None or mle or int

    # outer cv for score estimates
    opts["random_state"] = np.random.randint(1e4)
    opts["out_fold"] = "repeated"  # stratified, loo, repeated
    opts["n_out"] = 5
    opts["outer_score"] = "roc_auc"  # accuracy, roc_auc, f1_macro, f1_weighted

    # inner cv for hyperparam tuning
    opts["in_fold"] = "stratified"  # stratified, loo, repeated
    opts["n_in"] = 5
    opts["inner_score"] = "roc_auc"  # accuracy, roc_auc, f1_macro, f1_weighted

    # multiclass/label
    opts["multilabel"] = False
    opts["multiclass"] = False

    opts.update(kwargs)

    # gridsearch params
    opts["Cs"] = np.logspace(-4, 4, opts["n_lambda"])
    opts["alphas"] = np.linspace(0, 1, opts["n_alpha"])

    return opts
