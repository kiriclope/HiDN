import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearClassifierMixin, SparseCoefMixin
from sklearn.utils.extmath import softmax


class LinearSVCCV(LinearClassifierMixin, SparseCoefMixin, BaseEstimator):
    def __init__(
        self,
        Cs=[0.001, 0.01, 0.1, 1, 10, 100],
        cv=5,
        penalty="l1",
        loss="squared_hinge",
        dual=False,
        tol=1e-4,
        multi_class="ovr",
        fit_intercept=True,
        intercept_scaling=1,
        class_weight="balanced",
        verbose=0,
        random_state=None,
        max_iter=1000,
        n_jobs=None,
    ):
        self.Cs = Cs
        self.cv = cv

        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def fit(self, X, y):
        # base model
        self.svc_ = LinearSVC(
            penalty=self.penalty,
            loss=self.loss,
            dual=self.dual,
            tol=self.tol,
            multi_class=self.multi_class,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight,
            verbose=self.verbose,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )

        # grid
        self.grid_search = GridSearchCV(
            self.svc_, param_grid={"C": self.Cs}, cv=self.cv, n_jobs=self.n_jobs
        )

        # fit grid
        self.grid_search.fit(X, y)
        self.svc_ = self.grid_search.best_estimator_

        self.coef_ = self.svc_.coef_
        self.intercept_ = self.svc_.intercept_
        self.classes_ = self.svc_.classes_
        self.n_features_in_ = self.svc_.n_features_in_
        # self.feature_names_in_ = self.svc_.feature_names_in_
        self.n_iter_ = self.svc_.n_iter_

        return self

    def predict(self, X):
        return self.svc_.predict(X)

    def predict_proba(self, X):
        decision = self.svc_.decision_function(X)
        if decision.ndim == 1:
            # Workaround for multi_class="multinomial" and binary outcomes
            # which requires softmax prediction with only a 1D decision.
            decision_2d = np.c_[-decision, decision]
        else:
            decision_2d = decision

        return softmax(decision_2d, copy=False)

    def score(self, X, y, sample_weight=None):
        return self.svc_.score(X, y, sample_weight)
