import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


class SGDClassifierCV(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        cv=None,
        alphas=np.logspace(-4, 4, 10),
        l1_ratios=np.linspace(0, 1, 10),
        l1_ratio=0.15,
        loss="log_loss",
        penalty="l1",
        fit_intercept=True,
        max_iter=1000,
        tol=0.001,
        shuffle=True,
        verbose=0,
        epsilon=0.1,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=10,
        class_weight=None,
        warm_start=False,
        average=False,
    ):

        self.cv = cv
        self.alphas = alphas
        self.l1_ratios = l1_ratios
        self.l1_ratio = l1_ratio

        self.loss = loss
        self.penalty = penalty
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = average

    def fit(self, X, y):

        clf = SGDClassifier(
            l1_ratio=self.l1_ratio,
            loss=self.loss,
            penalty=self.penalty,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=self.shuffle,
            verbose=self.verbose,
            epsilon=self.epsilon,
            n_jobs=None,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            power_t=self.power_t,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            class_weight=self.class_weight,
            warm_start=self.warm_start,
            average=self.average,
        )

        if self.penalty == "elasticnet":
            param_grid = dict(alpha=self.alphas, l1_ratio=self.l1_ratios)
        else:
            param_grid = dict(alpha=self.alphas)

        grid = GridSearchCV(clf, param_grid=param_grid, cv=self.cv, n_jobs=self.n_jobs)

        grid.fit(X, y)

        self.model = grid.best_estimator_
        self.coef_ = self.model.coef_

        return self
