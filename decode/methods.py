import numpy as np

from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.model_selection import (
    StratifiedKFold,
    LeaveOneOut,
    RepeatedStratifiedKFold,
)

from .cross_temp_utils import temp_cross_val_score, permutation_test_temp_score


def outer_cv(
    clf,
    X,
    y,
    scaling="standard",
    n_out=10,
    folds="stratified",
    outer_score="accuracy",
    inner_score="deviance",
    random_state=None,
    n_jobs=-1,
    return_clf=0,
):

    # pipe = set_scaler(clf, scaling)
    if folds == "stratified":
        cv_outer = StratifiedKFold(
            n_splits=n_out, shuffle=True, random_state=random_state
        )  # outer cv loop for scoring
    if folds == "loo":
        cv_outer = LeaveOneOut()
    if folds == "repeated":
        cv_outer = RepeatedStratifiedKFold(
            n_splits=n_out, n_repeats=100, random_state=random_state
        )

    cv_scores = cross_val_score(
        clf, X, y, cv=cv_outer, scoring=outer_score, n_jobs=n_jobs, verbose=1
    )

    # cv_scores = glmnet_cv_loop(deepcopy(clf), X, y, cv=cv_outer, inner_score=inner_score, outer_score=outer_score, return_clf=0)

    return np.nanmean(cv_scores)


def outer_temp_cv(
    model,
    X_t_train,
    X_t_test,
    y,
    n_out=5,
    folds="stratified",
    n_repeats=100,
    outer_score="accuracy",
    random_state=None,
    n_jobs=None,
    IF_SHUFFLE=0,
):

    if IF_SHUFFLE:
        method = permutation_test_temp_score
        # folds = 'stratified'
    else:
        method = temp_cross_val_score

    if folds == "stratified":
        cv_outer = StratifiedKFold(
            n_splits=n_out, shuffle=True, random_state=random_state
        )  # outer cv loop for scoring
    if folds == "loo":
        cv_outer = LeaveOneOut()
    if folds == "repeated":
        cv_outer = RepeatedStratifiedKFold(
            n_splits=n_out, n_repeats=n_repeats, random_state=random_state
        )

    if X_t_train.ndim > 2:
        cv_score = []
        pval = []
        for i in range(X_t_train.shape[-1]):
            X_t_train_i = X_t_train[..., i]
            X_t_test_i = X_t_test[..., i]

            if IF_SHUFFLE:
                cv_score_i, _, pval_i = method(
                    model,
                    X_t_train_i,
                    X_t_test_i,
                    y,
                    cv=cv_outer,
                    scoring=outer_score,
                    n_jobs=n_jobs,
                )
                pval.append(pval_i)

            else:
                cv_score_i = method(
                    model,
                    X_t_train_i,
                    X_t_test_i,
                    y,
                    cv=cv_outer,
                    scoring=outer_score,
                    n_jobs=n_jobs,
                )

            cv_score.append(cv_score_i)
    else:
        if IF_SHUFFLE:
            cv_score, shuffle, pval = method(
                model,
                X_t_train,
                X_t_test,
                y,
                cv=cv_outer,
                scoring=outer_score,
                n_jobs=n_jobs,
            )

            print("cv_score", cv_score, "shuffle", shuffle.shape, "pval", pval)
        else:
            cv_score = method(
                model,
                X_t_train,
                X_t_test,
                y,
                cv=cv_outer,
                scoring=outer_score,
                n_jobs=n_jobs,
            )

            print("cv_score", cv_score.shape)

    if IF_SHUFFLE:
        return cv_score, pval
    else:
        return np.nanmean(cv_score, axis=-1), None
