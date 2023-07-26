from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import ttest_1samp, ttest_ind
from sklearn.base import BaseEstimator, ClassifierMixin

# import .progressbar as pg


def boots_coefs_parloop(clf, X, y, confidence=0.05):
    np.random.seed(None)
    clf_copy = deepcopy(clf)

    n_0 = np.sum(y == 0)
    n_1 = np.sum(y == 1)

    idx_trials = np.hstack(
        (np.random.randint(0, n_0, n_0), np.random.randint(n_0, X.shape[0], n_1))
    )

    X_sample = X[idx_trials]
    y_sample = y[idx_trials]

    clf_copy.fit(X_sample, y_sample)
    coefs = np.zeros(X.shape[-1])

    if "filter" in clf_copy.named_steps.keys():
        pval = clf_copy.named_steps["filter"].pvalues_
        coef = clf_copy.named_steps["clf"].coef_[0]
        idx = pval <= confidence
        coefs[idx] = coef
    else:
        coefs = clf_copy.named_steps["clf"].coef_[0]

    # print('coef', coefs.shape, 'non zero', np.sum(coef!=0) )

    return coefs


def bootstrap_coefs(clf, X, y, n_boots=1000, confidence=0.05, n_jobs=-1):
    # with pg.tqdm_joblib(pg.tqdm(desc='bootstrap coefs', total=n_boots)) as progress_bar:
    dum = Parallel(n_jobs=n_jobs)(
        delayed(boots_coefs_parloop)(clf, X, y, confidence=confidence)
        for _ in range(n_boots)
    )

    boots_coefs = np.array(dum)

    return boots_coefs


class bolasso(BaseEstimator, ClassifierMixin):
    def __init__(self, clf, n_boots=1000, confidence=0.05, n_jobs=None, verbose=0):
        self.clf = clf
        self.confidence = confidence
        self.n_boots = n_boots

        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        self.model_ = self.clf

        self.boots_coef_ = bootstrap_coefs(
            self.model_,
            X,
            y,
            n_boots=self.n_boots,
            confidence=self.confidence,
            n_jobs=self.n_jobs,
        )

        mean_coefs = np.nanmean(self.boots_coef_, axis=0)
        self.mean_coef_ = mean_coefs

        if self.verbose:
            print("boots_coefs", self.boots_coef_.shape, mean_coefs[:10])

        # _, self.p_val_ = ttest_ind(
        #     self.boots_coef_,
        #     np.zeros(self.boots_coef_.shape),
        #     axis=0,
        #     equal_var=False,
        #     nan_policy="omit",
        # )

        _, self.p_val_ = ttest_1samp(
            self.boots_coef_,
            0,
            axis=0,
            nan_policy="omit",
        )

        if self.verbose:
            print("p_val", self.p_val_.shape, self.p_val_[:5])
            print("coefs", mean_coefs[:5])

        self.fs_idx_ = self.p_val_ <= self.confidence

        if self.verbose:
            print("significant", np.sum(self.fs_idx_))

        self.coef_ = np.zeros(X.shape[-1])
        X_fs = X[:, self.fs_idx_]

        if self.verbose:
            print("X_fs", X_fs.shape)

        self.model_.named_steps["clf"].penalty = "l1"
        self.model_.named_steps["clf"].n_jobs = -1

        self.model_.fit(X_fs, y)

        if "filter" in self.model_.named_steps.keys():
            pval = self.model_.named_steps["filter"].pvalues_
            coef = self.model_.named_steps["clf"].coef_[0]
            # print('coefs', self.model_.named_steps['clf'].coef_.shape)
            idx = pval <= self.confidence

            coefs = self.coef_[self.fs_idx_]
            coefs[idx] = coef
            self.coef_[self.fs_idx_] = coefs

            print(
                "coef",
                self.coef_.shape,
                # "lambda",
                # self.model_.named_steps["clf"].C_,
                "non zero",
                np.sum(idx),
            )
        else:
            coef = self.model_.named_steps["clf"].coef_[0]
            self.coef_[self.fs_idx_] = coef

            print(
                "coef",
                coef.shape,
                # "lambda",
                # self.model_.named_steps["clf"].C_,
                "non zero",
                np.sum(self.fs_idx_),
            )

        self.intercept_ = self.model_.named_steps["clf"].intercept_

        if "scaler" in self.model_.named_steps.keys():
            # try:
            #     mean = self.model_.named_steps["scaler"].mean_
            # except:
            #     try:
            #         mean = self.model_.named_steps["scaler"].center_
            #     except:
            #         pass
            scale = self.model_.named_steps["scaler"].scale_

            if scale is None:
                scale = np.ones(self.coef_[self.fs_idx_].shape[0])

            # print(mean.shape, scale.shape, self.coef_.shape)

            # self.coef_[self.fs_idx_] = np.true_divide(self.coef_[self.fs_idx_], scale)
            # self.intercept_ -= np.dot(self.coef_[self.fs_idx_], mean)

        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)
