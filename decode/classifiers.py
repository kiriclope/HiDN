import numpy as np

from sklearn.feature_selection import f_classif, SelectPercentile, SelectFpr
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    RepeatedStratifiedKFold,
)
from sklearn.ensemble import BaggingClassifier
from .bolasso_sklearn import bolasso


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
        n_jobs=kwargs["n_jobs"],
        verbose=0,
    )

    pipe = []
    if kwargs["standardize"] == "standard":
        pipe.append(("scaler", StandardScaler()))
    if kwargs["standardize"] == "center":
        pipe.append(("scaler", StandardScaler(with_std=False)))
    if kwargs["standardize"] == "robust":
        pipe.append(("scaler", RobustScaler(unit_variance=1)))
    if kwargs["prescreen"]:
        pipe.append(("filter", SelectFpr(f_classif, alpha=kwargs["pval"])))

    pipe.append(("clf", clf))
    pipe = Pipeline(pipe)

    if "bolasso" in kwargs["clf_name"]:
        pipe = bolasso(
            pipe,
            n_boots=kwargs["n_boots"],
            confidence=kwargs["pval"],
            n_jobs=kwargs["n_jobs"],
            verbose=0,
        )

    if "bootstrap" in kwargs["clf_name"]:
        pipe = BaggingClassifier(
            pipe, n_estimators=kwargs["n_boots"], n_jobs=kwargs["n_jobs"]
        )

    # print(pipe)
    print(
        "SCALER",
        kwargs["standardize"],
        "PRESCREEN",
        kwargs["prescreen"],
        "METHOD",
        kwargs["clf_name"],
        "FOLDS",
        kwargs["in_fold"],
    )

    return pipe
