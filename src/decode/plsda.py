import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.calibration import CalibratedClassifierCV

class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls_ = None
        self.calibrated_ = None
    
    def fit(self, X, y, calibration=True, **calibration_params):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Initialize and fit the PLS model on the full dataset
        self.pls_ = PLSRegression(n_components=self.n_components)
        self.pls_.fit(X, y)
        # Fit a calibration model using the predictions from PLS
        if calibration:
            self.calibrated_ = CalibratedClassifierCV(base_estimator=self, cv='prefit', **calibration_params)
            self.calibrated_.fit(X, y)

        self.coef_ = self.get_coefficients()
        
        return self
    
    def get_coefficients(self):
        # Check if fit has been called
        check_is_fitted(self)
        # Return the coefficients from the PLS regression model
        # The pls_ object contains an attribute coef_ after fitting
        if self.pls_ is not None:
            return self.pls_.coef_
        else:
            raise ValueError("The model has not been fitted yet.")
    
    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        # Use the calibrated model to predict probabilities if available
        if self.calibrated_ is not None:
            return self.calibrated_.predict_proba(X)
        # If not calibrated, fallback to raw PLS regression predictions (not probabilities)
        return self.pls_.predict(X)
    
    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        # Use the calibrated model to make predictions if available
        if self.calibrated_ is not None:
            return self.calibrated_.predict(X)
        # If not calibrated, threshold on raw PLS regression predictions
        scores = self.pls_.predict(X)
        return (scores >= 0.5).astype(int)  # Threshold of 0.5 used for binary classification

    def decision_function(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        # Return the regression scores which is what the CalibratedClassifierCV expects
        return self.pls_.predict(X).ravel()
