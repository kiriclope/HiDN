import numpy as np
from scipy.stats import bootstrap, ttest_1samp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaggingClassifier


class bolasso(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        clf,
        penalty="l2",
        n_boots=1000,
        pval=0.05,
        confidence=0.05,
        n_jobs=None,
        verbose=0,
    ):
        self.model_ = clf
        self.penalty = penalty
        self.pval = pval
        self.confidence = confidence

        self.n_boots = n_boots
        self.n_jobs = n_jobs
        self.verbose = verbose

    def get_bag_coef(self):
        self.bag_coef_ = np.zeros((self.n_boots, self.n_feat))

        for i_boot in range(self.n_boots):
            model_boot = self.bag_.estimators_[i_boot]
            coef = model_boot.named_steps["clf"].coef_[0]

            if "filter" in self.model_.named_steps.keys():
                pval = model_boot.named_steps["filter"].pvalues_
                idx = pval <= self.pval
                self.bag_coef_[i_boot, idx] = coef
            elif "corr" in self.model_.named_steps.keys():
                idx = model_boot.named_steps["corr"].to_drop
                all_indices = set(range(self.n_feat))  # All indices
                idx_set = set(idx)  # Indices in idx
                remaining_indices = list(all_indices - idx_set)
                self.bag_coef_[i_boot, remaining_indices] = coef
            else:
                self.bag_coef_[i_boot] = model_boot.named_steps["clf"].coef_[0]

        if self.verbose:
            print("boots_coefs", self.bag_coef_.shape)

    def get_fs_idx(self):
        _, self.p_val_ = ttest_1samp(self.bag_coef_, 0, axis=0, nan_policy="omit")

        if self.verbose:
            print("p_val", self.p_val_.shape)

        self.fs_idx_ = self.p_val_ <= self.confidence

        # non parametric
        # conf_int = np.percentile(
        #     self.bag_coef_,
        #     [100 * self.confidence / 2, 100 * (1 - self.confidence / 2)],
        #     axis=0,
        # )

        # print(conf_int.shape)
        # self.fs_idx_ = (conf_int[0] > 0) | (conf_int[1] < 0)

        # Calculate the consistency of non zero elements
        # consistency = np.mean(np.abs(self.bag_coef_) >= 1e-5, 0)
        # print("consistency", consistency.shape)

        # Filter variables that are consistently non-zero
        # self.fs_idx_ = consistency > (1 - self.confidence)

        if self.verbose:
            print("significant", np.sum(self.fs_idx_))

    def fit_model(self, X, y):
        self.model_.named_steps["clf"].penalty = self.penalty
        self.model_.named_steps["clf"].n_jobs = -1

        X_fs = X[:, self.fs_idx_]
        if self.verbose:
            print("X_fs", X_fs.shape)

        self.model_.fit(X_fs, y)

    def get_coef(self):
        self.coef_ = np.zeros(self.n_feat)

        if "filter" in self.model_.named_steps.keys():
            pval = self.model_.named_steps["filter"].pvalues_
            idx = pval <= self.pval

            coefs = self.coef_[self.fs_idx_]
            coefs[idx] = self.model_.named_steps["clf"].coef_[0]
            self.coef_[self.fs_idx_] = coefs

        elif "corr" in self.model_.named_steps.keys():
            idx_drop = self.model_.named_steps["corr"].to_drop
            all_indices = set(range(np.sum(self.fs_idx_)))
            idx_keep = list(all_indices - set(idx_drop))

            coefs = self.coef_[self.fs_idx_]
            coefs[idx_keep] = self.model_.named_steps["clf"].coef_[0]
            self.coef_[self.fs_idx_] = coefs

        else:
            self.coef_[self.fs_idx_] = self.model_.named_steps["clf"].coef_[0]

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

    def fit(self, X, y):
        self.n_feat = X.shape[1]

        self.bag_ = BaggingClassifier(
            self.model_, n_estimators=self.n_boots, n_jobs=self.n_jobs
        )

        # fit bag
        self.bag_.fit(X, y)
        # get coefficients from each fit
        self.get_bag_coef()
        # get sighificant idx
        self.get_fs_idx()
        # fit model
        self.fit_model(X, y)
        # get coef
        self.get_coef()

        print(
            "samples",
            y.shape,
            "features",
            self.coef_.shape,
            "non zero",
            np.sum(self.fs_idx_),
        )

        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
