import numpy as np
from dual_data.decode.bolasso_sklearn import bolasso
from dual_data.decode.SGDClassifierCV import SGDClassifierCV
from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC

# from sklearn.pipeline import Pipeline


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
        )

    if kwargs["clf"] == "LinearSVC":
        clf = LinearSVC(
            penalty=kwargs["penalty"],
            loss="squared_hinge",
            dual=False,
            tol=kwargs["tol"],
            C=1,
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

    if kwargs["standardize"] == "custom":
        pipe.append(("scaler", CustomScaler()))
    if kwargs["standardize"] == "minmax":
        pipe.append(("scaler", MinMaxScaler()))
    if kwargs["standardize"] == "standard":
        pipe.append(("scaler", StandardScaler()))
    if kwargs["standardize"] == "center":
        pipe.append(("scaler", StandardScaler(with_std=False)))
    if kwargs["standardize"] == "robust":
        pipe.append(("scaler", RobustScaler(unit_variance=False)))
    if kwargs["prescreen"]:
        pipe.append(("filter", SelectFpr(f_classif, alpha=kwargs["pval"])))
    if kwargs["imbalance"]:
        pipe.append(("bal", SVMSMOTE(random_state=kwargs["random_state"])))
    if kwargs["pca"]:
        pipe.append(("pca", PCA(n_components=kwargs["n_comp"])))

    pipe.append(("clf", clf))
    pipe = Pipeline(pipe)

    if kwargs["method"] is not None:
        if "bolasso" in kwargs["method"]:
            pipe = bolasso(
                pipe,
                penalty=kwargs["bolasso_penalty"],
                n_boots=kwargs["n_boots"],
                confidence=kwargs["pval"],
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
            else:
                param_grid = dict(clf__C=kwargs["Cs"])

            pipe = GridSearchCV(
                pipe, param_grid=param_grid, cv=cv, n_jobs=kwargs["n_jobs"]
            )

    print("##########################################")
    print(
        "MODEL:",
        "SCALER",
        kwargs["standardize"],
        "IMBALANCE",
        kwargs["imbalance"],
        "PRESCREEN",
        kwargs["prescreen"],
        "PCA",
        kwargs["pca"],
        "METHOD",
        kwargs["method"],
        "FOLDS",
        kwargs["in_fold"],
        "CLF",
        kwargs["clf"],
    )

    return pipe
