import numpy as np
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    LeaveOneOut,
    RepeatedStratifiedKFold,
)

from sklearn.base import clone

from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm

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


def score_parloop(model, method, X_t_train, X_t_test, y, cv, scoring):

    score, perm_score, pval = method(
        model,
        X_t_train,
        X_t_test,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=None,
    )

    # print("score", score.shape)

    return score, perm_score, pval


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

    if folds == "stratified":
        cv = StratifiedKFold(n_splits=n_out, shuffle=True, random_state=random_state)
    elif folds == "loo":
        cv = LeaveOneOut()
    elif folds == "repeated":
        cv = RepeatedStratifiedKFold(
            n_splits=n_out, n_repeats=n_repeats, random_state=random_state
        )

    if IF_SHUFFLE:
        method = permutation_test_temp_score
    else:
        method = temp_cross_val_score

    if X_t_train.ndim > 2:

        with parallel_backend("dask"):
            scores, perm_scores, pvals = zip(
                *Parallel(n_jobs=n_jobs)(
                    delayed(score_parloop)(
                        clone(model),
                        method,
                        X_t_train[..., i],
                        X_t_test[..., i],
                        y,
                        cv=cv,
                        scoring=outer_score,
                    )
                    for i in tqdm(range(X_t_train.shape[-1]), desc="cv_score")
                )
            )

    else:
        scores, perm_scores, pvals = score_parloop(
            clone(model),
            method,
            X_t_train,
            X_t_test,
            y,
            cv=cv,
            scoring=outer_score,
        )

    scores = np.array(scores)
    perm_scores = np.array(perm_scores)
    pvals = np.array(pvals)

    print(
        "scores", scores.shape, "perm_scores", perm_scores.shape, "pvals", pvals.shape
    )
    return scores, perm_scores, pvals
