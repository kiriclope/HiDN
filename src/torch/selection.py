from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectPercentile, SelectKBest, SelectFdr, SelectFpr, SelectFwe, f_classif, VarianceThreshold

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
            self.selector = SelectKBest(f_classif, k=alpha * 100)
        elif 'perc' in method:
            self.selector = SelectPercentile(f_classif, percentile=alpha * 100)
        elif 'var' in method:
            self.selector = VarianceThreshold(threshold=alpha)

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
