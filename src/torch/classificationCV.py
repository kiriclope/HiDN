from time import perf_counter
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, LeaveOneOut
from sklearn.decomposition import PCA

from mne.decoding import SlidingEstimator, cross_val_multiscore

class ClassificationCV():
    def __init__(self, net, params, **kwargs):

        pipe = []
        self.scaler = kwargs['scaler']
        if self.scaler is not None and self.scaler !=0 :
            pipe.append(("scaler", StandardScaler()))

        self.n_comp = kwargs['n_comp']
        if kwargs['n_comp'] is not None:
            self.n_comp = kwargs['n_comp']
            pipe.append(("pca", PCA(n_components=self.n_comp)))

        pipe.append(("net", net))
        self.model = Pipeline(pipe)

        self.num_features = kwargs['num_features']
        self.scoring =  kwargs['scoring']

        if  kwargs['n_splits']==-1:
            self.cv = LeaveOneOut()
        else:
            self.cv = RepeatedStratifiedKFold(n_splits=kwargs['n_splits'], n_repeats=kwargs['n_repeats'])

        self.params = params
        self.verbose =  kwargs['verbose']
        self.n_jobs =  kwargs['n_jobs']

    def fit(self, X, y):
        start = perf_counter()
        if self.verbose:
            print('Fitting hyperparameters ...')

        self.model['net'].module__num_features = self.num_features
        grid = GridSearchCV(self.model, self.params, refit=True, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs)
        grid.fit(X.astype('float32'), y.astype('float32'))
        end = perf_counter()
        if self.verbose:
            print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))

        self.best_model = grid.best_estimator_
        self.best_params = grid.best_params_

        if self.verbose:
            print(self.best_params)

        self.coefs = self.best_model.named_steps['net'].module_.linear.weight.data.cpu().detach().numpy()[0]
        self.bias = self.best_model.named_steps['net'].module_.linear.bias.data.cpu().detach().numpy()[0]

    def get_bootstrap_coefs(self, X, y, n_boots=10):
        start = perf_counter()
        if self.verbose:
            print('Bootstrapping coefficients ...')

        self.bagging_clf = BaggingClassifier(base_estimator=self.best_model, n_estimators=n_boots)
        self.bagging_clf.fit(X.astype('float32'), y.astype('float32'))
        end = perf_counter()

        if self.verbose:
            print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))

        self.coefs, self.bias = get_bagged_coefs(self.bagging_clf, n_estimators=n_boots)

        return self.coefs, self.bias

    def get_overlap(self, model, X):
        coefs = model.named_steps['net'].module_.linear.weight.data.cpu().detach().numpy()[0]
        bias = model.named_steps['net'].module_.linear.bias.data.cpu().detach().numpy()[0]

        if self.scaler is not None and self.scaler!=0:
            scaler = model.named_steps['scaler']
            for i in range(X.shape[-1]):
                X[..., i] = scaler.transform(X[..., i])

        if self.n_comp is not None:
            pca = model.named_steps['pca']
            X_pca = np.zeros((X.shape[0], self.n_comp, X.shape[-1]))

            for i in range(X.shape[-1]):
                X_pca[..., i] = pca.transform(X[..., i])

            self.overlaps = (np.swapaxes(X_pca, 1, -1) @ coefs + bias) / np.linalg.norm(coefs)
        else:
            self.overlaps = -(np.swapaxes(X, 1, -1) @ coefs + bias) / np.linalg.norm(coefs)

        return self.overlaps

    def get_bootstrap_overlaps(self, X):
        start = perf_counter()
        if self.verbose:
            print('Getting bootstrapped overlaps ...')

        X_copy = np.copy(X)
        overlaps_list = []
        n_boots = len(self.bagging_clf.estimators_)

        for i in range(n_boots):
            model = self.bagging_clf.estimators_[i]
            overlaps = self.get_overlap(model, X_copy)
            overlaps_list.append(overlaps)

        end = perf_counter()
        if self.verbose:
            print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))

        return np.array(overlaps_list).mean(0)

    def get_cv_scores(self, X, y, scoring):
        start = perf_counter()
        if self.verbose:
            print('Computing cv scores ...')

        estimator = SlidingEstimator(clone(self.best_model), n_jobs=1,
                                     scoring=scoring, verbose=False)

        self.scores = cross_val_multiscore(estimator, X.astype('float32'), y.astype('float32'),
                                           cv=self.cv, n_jobs=-1, verbose=False)
        end = perf_counter()
        if self.verbose:
            print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))

        return self.scores
