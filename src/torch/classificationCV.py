import copy
from time import perf_counter
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, LeaveOneOut
from sklearn.decomposition import PCA

from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore
from src.decode.my_mne import my_cross_val_multiscore, my_cross_val_compo_score


def convert_seconds(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return h, m, s


class ClassificationCV:
    def __init__(self, net, params, **kwargs):
        pipe = []
        self.scaler = kwargs["scaler"]
        if self.scaler is not None and self.scaler != 0:
            pipe.append(("scaler", StandardScaler()))

        self.n_comp = kwargs["n_comp"]
        if kwargs["n_comp"] is not None:
            self.n_comp = kwargs["n_comp"]
            pipe.append(("pca", PCA(n_components=self.n_comp)))

        pipe.append(("net", net))
        self.model = Pipeline(pipe)

        self.hp_scoring = kwargs["hp_scoring"]
        self.scoring = kwargs["scoring"]

        if kwargs["n_splits"] == -1:
            self.cv = LeaveOneOut()
        else:
            self.cv = RepeatedStratifiedKFold(
                n_splits=kwargs["n_splits"], n_repeats=kwargs["n_repeats"]
            )

        self.params = params
        self.verbose = kwargs["verbose"]
        self.n_jobs = kwargs["n_jobs"]

    def fit(self, X, y):
        start = perf_counter()
        if self.verbose:
            print("Fitting hyperparameters ...")

        grid = GridSearchCV(
            self.model,
            self.params,
            refit=True,
            cv=self.cv,
            scoring=self.hp_scoring,
            n_jobs=self.n_jobs,
        )
        grid.fit(X.astype("float32"), y.astype("float32"))
        end = perf_counter()
        if self.verbose:
            print(
                "Elapsed (with compilation) = %dh %dm %ds"
                % convert_seconds(end - start)
            )

        self.best_model = grid.best_estimator_
        self.best_params = grid.best_params_

        if self.verbose:
            print(self.best_params)

        try:
            self.coefs = (
                self.best_model.named_steps["net"]
                .module_.linear.weight.data.cpu()
                .detach()
                .numpy()[0]
            )
            self.bias = (
                self.best_model.named_steps["net"]
                .module_.linear.bias.data.cpu()
                .detach()
                .numpy()[0]
            )
        except:
            self.coefs = self.best_model.named_steps["net"].coef_.T
            self.bias = self.best_model.named_steps["net"].intercept_.T

    def get_bootstrap_coefs(self, X, y, n_boots=10):
        start = perf_counter()
        if self.verbose:
            print("Bootstrapping coefficients ...")

        self.bagging_clf = BaggingClassifier(
            base_estimator=self.best_model, n_estimators=n_boots
        )
        self.bagging_clf.fit(X.astype("float32"), y.astype("float32"))
        end = perf_counter()

        if self.verbose:
            print(
                "Elapsed (with compilation) = %dh %dm %ds"
                % convert_seconds(end - start)
            )

        self.coefs, self.bias = get_bagged_coefs(self.bagging_clf, n_estimators=n_boots)

        return self.coefs, self.bias

    def get_overlap(self, model, X):
        try:
            coefs = (
                model.named_steps["net"]
                .module_.linear.weight.data.cpu()
                .detach()
                .numpy()[0]
            )
            bias = (
                model.named_steps["net"]
                .module_.linear.bias.data.cpu()
                .detach()
                .numpy()[0]
            )
        except:
            coefs = model.named_steps["net"].coef_.T
            bias = model.named_steps["net"].intercept_.T

        if self.scaler is not None and self.scaler != 0:
            scaler = model.named_steps["scaler"]
            for i in range(X.shape[-1]):
                X[..., i] = scaler.transform(X[..., i])

        if self.n_comp is not None:
            pca = model.named_steps["pca"]
            X_pca = np.zeros((X.shape[0], self.n_comp, X.shape[-1]))

            for i in range(X.shape[-1]):
                X_pca[..., i] = pca.transform(X[..., i])

            self.overlaps = (
                np.swapaxes(X_pca, 1, -1) @ coefs + bias
            )  # / np.linalg.norm(coefs, axis=0)
        else:
            self.overlaps = -(
                np.swapaxes(X, 1, -1) @ coefs + bias
            )  # / np.linalg.norm(coefs, axis=0)

        return self.overlaps

    def get_bootstrap_overlaps(self, X):
        start = perf_counter()
        if self.verbose:
            print("Getting bootstrapped overlaps ...")

        X_copy = np.copy(X)
        overlaps_list = []
        n_boots = len(self.bagging_clf.estimators_)

        for i in range(n_boots):
            model = self.bagging_clf.estimators_[i]
            overlaps = self.get_overlap(model, X_copy)
            overlaps_list.append(overlaps)

        end = perf_counter()
        if self.verbose:
            print(
                "Elapsed (with compilation) = %dh %dm %ds"
                % convert_seconds(end - start)
            )

        return np.array(overlaps_list).mean(0)

    def get_cv_scores(self, X, y, scoring, cv=None, X_test=None, y_test=None, IF_GEN=0, IF_COMPO=0):
        if cv is None:
            cv = self.cv
        if X_test is None:
            X_test = X
            y_test = y
            print('X_test==X', True)

        start = perf_counter()
        if self.verbose:
            print("Computing cv scores ...")

        if IF_GEN==0:
            self.estimator = SlidingEstimator(
                copy.deepcopy(self.best_model), n_jobs=1, scoring=scoring, verbose=False
            )
        else:
            self.estimator = GeneralizingEstimator(copy.deepcopy(self.best_model), n_jobs=1, scoring=scoring, verbose=False)

        # self.scores = cross_val_multiscore(estimator, X.astype('float32'), y.astype('float32'),
        #                                    cv=cv, n_jobs=-1, verbose=False)
        if IF_COMPO:
            self.scores = my_cross_val_compo_score(
                self.estimator,
                X.astype("float32"),
                X_test.astype("float32"),
                y.astype("float32"),
                y_test,
                cv=cv,
                n_jobs=-1,
                verbose=False,
            )
        else:
            self.scores = my_cross_val_multiscore(
                self.estimator,
                X.astype("float32"),
                X_test.astype("float32"),
                y.astype("float32"),
                y_test,
                cv=cv,
                n_jobs=-1,
                verbose=False,
            )

        end = perf_counter()
        if self.verbose:
            print(
                "Elapsed (with compilation) = %dh %dm %ds"
                % convert_seconds(end - start)
            )

        return self.scores
