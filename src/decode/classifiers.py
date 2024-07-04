import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SVMSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from scipy.spatial.distance import correlation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectPercentile, SelectKBest, SelectFdr, SelectFpr, SelectFwe, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC
from mne.decoding import Scaler, Vectorizer

from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch import NeuralNetClassifier

from src.decode.bolasso_sklearn import bolasso
from src.decode.SGDClassifierCV import SGDClassifierCV
from src.decode.LinearSVCCV import LinearSVCCV
from src.decode.perceptron import Perceptron, RegularizedNet

# from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr


class PearsonCorrSelector(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.05, correction_method='bonferroni'):
        self.alpha = alpha
        self.support_ = None
        self.correction_method = correction_method

    def fit(self, X, y):
        p_values = np.array([pearsonr(X[:, i], y)[1] for i in range(X.shape[1])])
        if self.correction_method == 'bonferroni':
            # Bonferroni correction
            corrected_alpha = self.alpha / len(p_values)
        elif self.correction_method is None:
            # No correction
            corrected_alpha = self.alpha
        else:
            raise ValueError("Unsupported correction method provided: {}".format(self.correction_method))

        self.support_ = p_values <= corrected_alpha
        return self

    def transform(self, X):
        return X[:, self.support_]

    def _get_support_mask(self):
        if self.support_ is None:
            raise ValueError("The fit method has not been called yet.")
        return self.support_

    def get_support(self, indices=False):
        if indices:
            return np.where(self._get_support_mask())[0]
        return self._get_support_mask()

class safeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method='fpr', alpha=0.05):
        self.method = method
        self.alpha = alpha

        if 'fpr' in method:
            self.selector = SelectFpr(f_classif, alpha=alpha)
        elif 'fdr' in method:
            self.selector = SelectFdr(f_classif, alpha=alpha)
        elif 'Fwe' in method:
            self.selector = SelectFwe(f_classif, alpha=alpha)
        elif 'kbest' in method:
            self.selector = SelectKBest(f_classif, k=alpha)
        elif 'perc' in method:
            self.selector = SelectPercentile(f_classif, percentile=alpha * 100)
        elif 'var' in method:
            self.selector = VarianceThreshold(threshold=alpha)
        elif 'corr' in method:
            self.selector = PearsonCorrSelector(alpha=alpha)

        self.feature_indices_ = None

    def fit(self, X, y=None):
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        X_t = self.selector.transform(X)
        if X_t.shape[1] == 0:
            self.feature_indices_ = [0]  # fallback to the first feature
            return X[:, [0]]
        else:
            self.feature_indices_ = self.selector.get_support(indices=True)
            return X_t

    def _get_support_mask(self):
        return self.selector._get_support_mask()

class CorrelationThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        corr_mat = df.corr().abs()
        upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool_))
        self.to_drop = [
            column for column in upper.columns if any(upper[column] > self.threshold)
        ]
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X)
        return df.drop(df.columns[self.to_drop], axis=1)


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler_1 = MinMaxScaler()
        self.scaler_0 = MinMaxScaler()

    def fit(self, X, y=None):
        X_1 = X[y == 1]
        X_0 = X[y == 0]

        print("X1", X_1.shape)

        self.scaler_1.fit(X_1)
        self.scaler_0.fit(X_0)
        return self

    def transform(self, X, y=None):
        X_1 = X[y == 1]
        X_0 = X[y == 0]

        print("X", X.shape, "X1", X_1.shape)

        scaled_1 = self.scaler_1.transform(X_1)
        scaled_0 = self.scaler_0.transform(X_0) * -1
        return np.concatenate((scaled_1, scaled_0), axis=0)


def get_clf(**kwargs):
    if kwargs["in_fold"] == "stratified":
        cv = StratifiedKFold(
            n_splits=kwargs["n_in"], shuffle=True, random_state=kwargs["random_state"]
        )  # outer cv loop for scoring

    if kwargs["in_fold"] == "loo":
        cv = LeaveOneOut()

    if kwargs["in_fold"] == "repeated":
        cv = RepeatedStratifiedKFold(
            n_splits=kwargs["n_in"],
            n_repeats=kwargs["n_repeats"],
            random_state=kwargs["random_state"],
        )

    if kwargs["penalty"] != "elasticnet":
        kwargs["l1_ratios"] = None
    else:
        kwargs["l1_ratios"] = np.linspace(0, 1, kwargs["n_alpha"])
        kwargs["solver"] = "saga"

    if kwargs["clf"] == "perceptron":
        early_stopping = EarlyStopping(
            monitor='valid_loss', # Metric to monitor
            patience=5, # Number of epochs to wait for improvement
            threshold=0.001, # Minimum change to qualify as an improvement
            threshold_mode='rel', # 'rel' for relative change
            lower_is_better=True # Set to True if lower metric values are better
        )

        clf = RegularizedNet(
            module=Perceptron,
            module__num_features=kwargs['num_feat'],
            module__dropout_rate=0.5,
            criterion=nn.BCEWithLogitsLoss,
            optimizer=optim.Adam,
            optimizer__lr=0.1,
            max_epochs=1000,
            callbacks=[early_stopping],
            verbose=0,
            # train_split=None,
            # iterator_train__shuffle=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

    if kwargs["clf"] == "log_loss":
        clf = LogisticRegressionCV(
            Cs=kwargs["Cs"],
            solver=kwargs["solver"],
            penalty=kwargs["penalty"],
            l1_ratios=kwargs["l1_ratios"],
            tol=kwargs["tol"],
            max_iter=int(kwargs["max_iter"]),
            scoring=kwargs["inner_score"],
            fit_intercept=kwargs["fit_intercept"],
            intercept_scaling=kwargs["intercept_scaling"],
            cv=cv,
            class_weight=kwargs["class_weight"],
            refit=kwargs["refit"],
            multi_class=kwargs["multi_class"],
            n_jobs=None,
            verbose=0,
            random_state=kwargs["random_state"],
        )

    if kwargs["clf"] == "SVM":
        clf = LinearSVCCV(
            Cs=kwargs["Cs"],
            cv=cv,
            penalty=kwargs["penalty"],
            loss="squared_hinge",
            dual=False,
            tol=kwargs["tol"],
            multi_class="ovr",
            fit_intercept=kwargs["fit_intercept"],
            intercept_scaling=kwargs["intercept_scaling"],
            class_weight=kwargs["class_weight"],
            verbose=0,
            random_state=kwargs["random_state"],
            max_iter=kwargs["max_iter"],
        )

    if kwargs["clf"] == "LDA":
        clf = LinearDiscriminantAnalysis(
            tol=kwargs["tol"], solver="lsqr", shrinkage=kwargs["shrinkage"]
        )

    if kwargs["clf"] == "SGD":
        clf = SGDClassifier(
            loss="log_loss",
            penalty=kwargs["penalty"],
            alpha=kwargs["alpha"],
            l1_ratio=kwargs["l1_ratio"],
            fit_intercept=True,
            max_iter=kwargs["max_iter"],
            tol=kwargs["tol"],
            shuffle=True,
            verbose=0,
            epsilon=0.1,
            n_jobs=None,
            random_state=None,
            learning_rate=kwargs["learning_rate"],
            eta0=0.0,
            power_t=0.5,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=10,
            class_weight=kwargs["class_weight"],
            warm_start=False,
            average=False,
        )

    if kwargs["clf"] == "SGDCV":
        clf = SGDClassifierCV(
            cv=cv,
            loss="log_loss",
            penalty=kwargs["penalty"],
            alphas=kwargs["Cs"],
            l1_ratios=kwargs["l1_ratios"],
            l1_ratio=kwargs["l1_ratio"],
            fit_intercept=True,
            max_iter=kwargs["max_iter"],
            tol=kwargs["tol"],
            shuffle=True,
            verbose=0,
            epsilon=0.1,
            n_jobs=None,
            random_state=None,
            learning_rate=kwargs["learning_rate"],
            eta0=0.0,
            power_t=0.5,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=10,
            class_weight=kwargs["class_weight"],
            warm_start=False,
            average=False,
        )

    pipe = []

    # resampling data
    if kwargs['balance'] == 'under':
        pipe.append(("sampler", RandomUnderSampler(random_state=kwargs['random_state'])))
    elif kwargs['balance'] == 'over':
        pipe.append(("sampler", RandomOverSampler(random_state=kwargs['random_state'])))
    elif kwargs['balance'] == 'aug':
        # data augmantation
        pipe.append(("sampler", SVMSMOTE(random_state=kwargs["random_state"])))

    # scaling
    if kwargs["scaler"] == "custom":
        pipe.append(("scaler", CustomScaler()))
    if kwargs["scaler"] == "minmax":
        pipe.append(("scaler", MinMaxScaler()))
    if kwargs["scaler"] == "standard":
        pipe.append(("scaler", StandardScaler()))
    if kwargs["scaler"] == "mne_standard":
        pipe.append(("scaler", Scaler(scalings="mean")))
        pipe.append(("vec", Vectorizer()))
    if kwargs["scaler"] == "center":
        pipe.append(("scaler", StandardScaler(with_std=False)))
    if kwargs["scaler"] == "median":
        pipe.append(("scaler", RobustScaler(with_scaling=False)))
    if kwargs["scaler"] == "robust":
        pipe.append(("scaler", RobustScaler(unit_variance=kwargs["unit_var"])))
    if kwargs["scaler"] == "mne_robust":
        pipe.append(("scaler", Scaler(scalings="median")))
        pipe.append(("vec", Vectorizer()))

    # prescreen
    if kwargs["prescreen"] is not None:
        pipe.append(("filter", safeSelector(method=kwargs['prescreen'] , alpha=kwargs["pval"])))

    # dim red
    if kwargs["pca"]:
        pipe.append(("pca", PCA(n_components=kwargs["n_comp"])))
    if kwargs["corr"]:
        pipe.append(("corr", CorrelationThreshold(threshold=kwargs["threshold"])))

    pipe.append(("clf", clf))
    pipe = Pipeline(pipe)

    if kwargs["method"] is not None:
        if "bolasso" in kwargs["method"]:
            pipe = bolasso(
                pipe,
                penalty=kwargs["bolasso_penalty"],
                n_boots=kwargs["n_boots"],
                pval=kwargs["pval"],
                confidence=kwargs["bolasso_pval"],
                n_jobs=kwargs["n_jobs"],
                verbose=1,
            )

        if "bootstrap" in kwargs["method"]:
            pipe = BaggingClassifier(
                pipe, n_estimators=kwargs["n_boots"], n_jobs=kwargs["n_jobs"]
            )

        if "gridsearch" in kwargs["method"]:
            if kwargs["clf"] == "SGD":
                param_grid = dict(
                    clf__alpha=kwargs["Cs"], clf__l1_ratio=kwargs["l1_ratios"]
                )
            elif kwargs["clf"] == "perceptron":

                param_grid = {
                    'clf__alpha': np.logspace(-4, 4, 10),
                    'clf__l1_ratio': np.linspace(0, 1, 10),
                    'clf__module__dropout_rate': np.linspace(0, 1, 10),
                }

            else:
                param_grid = dict(clf__C=kwargs["Cs"])

            pipe = GridSearchCV(
                pipe, param_grid=param_grid, cv=cv, n_jobs=kwargs["n_jobs"]
            )

    print("##########################################")
    print(
        "MODEL:",
        kwargs['clf'],
        "FOLDS",
        kwargs['in_fold'],
        "RESAMPLE",
        kwargs["balance"],
        "SCALER",
        kwargs["scaler"],
        "PRESCREEN",
        kwargs["prescreen"],
        "PCA",
        kwargs["pca"],
        "METHOD",
        kwargs["method"],
    )

    # print(pipe)
    return pipe
