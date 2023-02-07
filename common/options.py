import multiprocessing
import numpy as np


def set_options(**kwargs):

    opts = dict()

    opts["reload"] = False
    opts["n_jobs"] = int(0.5 * multiprocessing.cpu_count())
    opts["verbose"] = 0

    ################################
    # task param
    ################################
    opts["tasks"] = np.array(["DPA", "DualGo", "DualNoGo"])

    opts["trials"] = "correct"  # 'correct', 'incorrect', ''
    opts["features"] = "sample"  # 'sample', 'distractor', 'task', 'reward'

    opts["task"] = "DualGo"  # DPA, DualGo, DualNoGo, Dual, or all
    opts["day"] = "first"  # int or 'first', 'middle', 'last'
    opts["laser"] = 0

    opts["overlap"] = "Dist"

    ################################
    # perm/boots param
    ################################
    opts["n_samples"] = 1000  # for permutation test
    opts["n_shuffles"] = 1000  # for permutation test
    opts["n_boots"] = 1000  # for bootstrap

    opts["ci"] = 1
    opts["shuffle"] = 1
    opts["perm_test"] = 1

    opts["add_vlines"] = 0

    ################################
    # preprocessing parameters
    ################################
    opts["epochs"] = ["ED", "MD", "LD"]
    opts["epoch"] = ["ED"]

    # scaling fluo
    opts["scaler_BL"] = "robust"  # "robust"  # standard, robust, center
    opts["center_BL"] = None
    opts["scale_BL"] = None
    opts["avg_mean_BL"] = 1
    opts["avg_noise_BL"] = 1
    opts["unit_var_BL"] = 1
    opts["return_center_scale"] = 0

    ################################
    # Synthetic augmentation
    ################################
    # create synthetic traces with spawner
    opts["augment"] = False
    opts["sig_aug"] = 0.005
    opts["n_aug"] = 1

    # adjust imbalance in trial types
    opts["imbalance"] = False

    ################################
    # classification parameters
    ################################

    opts["clf_name"] = "bootstrap"  # '' or 'bolasso' or 'bootstrap'

    opts["tol"] = 1e-3
    opts["max_iter"] = int(1e4)
    opts["fit_intercept"] = True
    opts["intercept_scaling"] = 1

    # penalty
    opts["penalty"] = "l1"
    opts["solver"] = "liblinear"  # liblinear or saga

    opts["n_lambda"] = 40
    opts["alpha"] = 0.5  # between 0 and 1
    opts["n_alpha"] = 10

    # standardization for classification
    opts["standardize"] = None  # 'standard', 'robust', 'center', None

    # prescreening
    opts["prescreen"] = False
    opts["pval"] = 0.05

    # outer cv for score estimates
    opts["random_state"] = None
    opts["out_fold"] = "loo"  # stratified, loo, repeated
    opts["n_out"] = 5
    opts["outer_score"] = "f1_weighted"  # accuracy, roc_auc, f1_macro

    # inner cv for hyperparam tuning
    opts["in_fold"] = "stratified"
    opts["n_in"] = 5
    opts["n_repeats"] = 10
    opts["inner_score"] = "f1_weighted"  # accuracy

    # multiclass/label
    opts["multilabel"] = False
    opts["multiclass"] = False

    opts.update(kwargs)
    opts["Cs"] = np.logspace(-4, 4, opts["n_lambda"])
    opts["alphas"] = np.linspace(0, 1, opts["n_alpha"])

    return opts
