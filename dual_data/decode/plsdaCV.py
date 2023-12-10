from dual_data.decode.plsda import PLSDAClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class PLSDAClassifierCV:
    def __init__(self, param_grid, cv=5, scoring=None):
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y):        
        # Create a GridSearchCV instance
        grid_search = GridSearchCV(
            estimator=PLSDAClassifier(),
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cv
        )
        
        # Perform the grid search
        self.grid_search = grid_search.fit(X, y)
        self.best_estimator_ = grid_search.best_estimator_
        self.best_score_ = grid_search.best_score_
        self.best_params_ = grid_search.best_params_
        self.cv_results_ = grid_search.cv_results_
        self.coef_= grid_search.best_estimator_.coef_
        
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)
