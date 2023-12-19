#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from common.get_data import get_X_y_days, get_X_y_S1_S2
from common.options import set_options
from common.plot_utils import add_vlines
from decode.classifiers import get_clf
from decode.methods import outer_temp_cv
from mne.decoding import (
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
)
from preprocess.helpers import avg_epochs, preprocess_X
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)


def get_score(model, X_train, X_test, y, **options):
    if X_train.ndim > 2:
        cv_score = []
        for i in range(X_train.shape[-1]):
            X_train_i = X_train[..., i]
            X_test_i = X_test[..., i]

            cv_score.append(
                outer_temp_cv(
                    model,
                    X_train_i,
                    X_test_i,
                    y,
                    n_out=options["n_out"],
                    folds=options["out_fold"],
                    outer_score=options["outer_score"],
                    random_state=None,
                    n_jobs=options["n_jobs"],
                )
            )
    else:
        cv_score = outer_temp_cv(
            model,
            X_train,
            X_test,
            y,
            n_out=options["n_out"],
            folds=options["out_fold"],
            inner_score=options["inner_score"],
            outer_score=options["outer_score"],
            random_state=None,
            n_jobs=options["n_jobs"],
        )

    return cv_score


if __name__ == "__main__":
    options = set_options()
    # X_days, y_days = get_X_y_days(IF_PREP=1, IF_AVG=0)
    X_days, y_days = get_X_y_days(**options)

    # X_days = preprocess_X(
    #     X_days,
    #     scaler=options["scaler_BL"],
    #     avg_mean=options["avg_mean_BL"],
    #     avg_noise=options["avg_noise_BL"],
    #     unit_var=options["unit_var_BL"],
    # )

    model = get_clf(**options)

    X_S1_S2, y_S1_S2 = get_X_y_S1_S2(X_days, y_days, **options)
    print(X_S1_S2.shape, y_S1_S2.shape)

    # cv_score = get_score(model, X_S1_S2, X_S1_S2, y_S1_S2, **options)
    # print("cv_score", cv_score.shape)

    time_decod = SlidingEstimator(
        model, n_jobs=-1, scoring=options["outer_score"], verbose=False
    )

    cv = RepeatedStratifiedKFold(
        n_splits=options["n_in"],
        n_repeats=options["n_repeats"],
        random_state=options["random_state"],
    )

    # here we use cv=3 just for speed
    scores = cross_val_multiscore(time_decod, X_S1_S2, y_S1_S2, cv=cv, n_jobs=-1)
    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    # Plot
    time = np.linspace(0, 14, int(14 * 6))

    fig = plt.figure("score")
    ax = plt.gca()
    plt.plot(time, scores, label="score")
    ax.axhline(0.5, color="k", linestyle="--", label="chance")
    plt.xlabel("Time (s)")
    plt.ylabel("Score")
    add_vlines()
    plt.ylim([0.25, 1])
    plt.yticks([0.25, 0.5, 0.75, 1])
