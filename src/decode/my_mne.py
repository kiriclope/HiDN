"""Base class copy from sklearn.base."""
# Authors: Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Romain Trachel <trachelr@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD-3-Clause

import datetime as dt
import numbers
import copy
import numpy as np
from mne.fixes import BaseEstimator, _get_check_scoring, is_classifier
from mne.parallel import parallel_func
from mne.utils import verbose, warn
from sklearn.model_selection import LeaveOneOut


class LinearModel(BaseEstimator):
    """Compute and store patterns from linear models.

    The linear model coefficients (filters) are used to extract discriminant
    neural sources from the measured data. This class computes the
    corresponding patterns of these linear filters to make them more
    interpretable :footcite:`HaufeEtAl2014`.

    Parameters
    ----------
    model : object | None
        A linear model from scikit-learn with a fit method
        that updates a ``coef_`` attribute.
        If None the model will be LogisticRegression.

    Attributes
    ----------
    filters_ : ndarray, shape ([n_targets], n_features)
        If fit, the filters used to decompose the data.
    patterns_ : ndarray, shape ([n_targets], n_features)
        If fit, the patterns used to restore M/EEG signals.

    See Also
    --------
    CSP
    mne.preprocessing.ICA
    mne.preprocessing.Xdawn

    Notes
    -----
    .. versionadded:: 0.10

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, model=None):  # noqa: D102
        if model is None:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(solver="liblinear")

        self.model = model
        self._estimator_type = getattr(model, "_estimator_type", None)

    def fit(self, X, y, **fit_params):
        """Estimate the coefficients of the linear model.

        Save the coefficients in the attribute ``filters_`` and
        computes the attribute ``patterns_``.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The training input samples to estimate the linear coefficients.
        y : array, shape (n_samples, [n_targets])
            The target values.
        **fit_params : dict of string -> object
            Parameters to pass to the fit method of the estimator.

        Returns
        -------
        self : instance of LinearModel
            Returns the modified instance.
        """
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 2:
            raise ValueError(
                "LinearModel only accepts 2-dimensional X, got "
                "%s instead." % (X.shape,)
            )
        if y.ndim > 2:
            raise ValueError(
                "LinearModel only accepts up to 2-dimensional y, "
                "got %s instead." % (y.shape,)
            )

        # fit the Model
        self.model.fit(X, y, **fit_params)

        # Computes patterns using Haufe's trick: A = Cov_X . W . Precision_Y

        inv_Y = 1.0
        X = X - X.mean(0, keepdims=True)
        if y.ndim == 2 and y.shape[1] != 1:
            y = y - y.mean(0, keepdims=True)
            inv_Y = np.linalg.pinv(np.cov(y.T))
        self.patterns_ = np.cov(X.T).dot(self.filters_.T.dot(inv_Y)).T

        return self

    @property
    def filters_(self):
        if hasattr(self.model, "coef_"):
            # Standard Linear Model
            filters = self.model.coef_
        elif hasattr(self.model.best_estimator_, "coef_"):
            # Linear Model with GridSearchCV
            filters = self.model.best_estimator_.coef_
        else:
            raise ValueError("model does not have a `coef_` attribute.")
        if filters.ndim == 2 and filters.shape[0] == 1:
            filters = filters[0]
        return filters

    def transform(self, X):
        """Transform the data using the linear model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            The predicted targets.
        """
        return self.model.transform(X)

    def fit_transform(self, X, y):
        """Fit the data and transform it using the linear model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The training input samples to estimate the linear coefficients.
        y : array, shape (n_samples,)
            The target values.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            The predicted targets.
        """
        return self.fit(X, y).transform(X)

    def predict(self, X):
        """Compute predictions of y from X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data used to compute the predictions.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            The predictions.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """Compute probabilistic predictions of y from X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data used to compute the predictions.

        Returns
        -------
        y_pred : array, shape (n_samples, n_classes)
            The probabilities.
        """
        return self.model.predict_proba(X)

    def decision_function(self, X):
        """Compute distance from the decision function of y from X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data used to compute the predictions.

        Returns
        -------
        y_pred : array, shape (n_samples, n_classes)
            The distances.
        """
        return self.model.decision_function(X)

    def score(self, X, y):
        """Score the linear model computed on the given test data.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data to transform.
        y : array, shape (n_samples,)
            The target values.

        Returns
        -------
        score : float
            Score of the linear model.
        """
        return self.model.score(X, y)

    # Needed for sklearn 1.3+
    @property
    def classes_(self):
        """The classes (pass-through to model)."""
        return self.model.classes_


def _set_cv(cv, estimator=None, X=None, y=None):
    """Set the default CV depending on whether clf is classifier/regressor."""
    # Detect whether classification or regression
    if estimator in ["classifier", "regressor"]:
        est_is_classifier = estimator == "classifier"
    else:
        est_is_classifier = is_classifier(estimator)
    # Setup CV
    from sklearn import model_selection as models
    from sklearn.model_selection import KFold, StratifiedKFold, check_cv

    if isinstance(cv, (int, np.int64)):
        XFold = StratifiedKFold if est_is_classifier else KFold
        cv = XFold(n_splits=cv)
    elif isinstance(cv, str):
        if not hasattr(models, cv):
            raise ValueError("Unknown cross-validation")
        cv = getattr(models, cv)
        cv = cv()
    cv = check_cv(cv=cv, y=y, classifier=est_is_classifier)

    # Extract train and test set to retrieve them at predict time
    cv_splits = [(train, test) for train, test in cv.split(X=np.zeros_like(y), y=y)]

    if not np.all([len(train) for train, _ in cv_splits]):
        raise ValueError("Some folds do not have any train epochs.")

    return cv, cv_splits


def _check_estimator(estimator, get_params=True):
    """Check whether an object has the methods required by sklearn."""
    valid_methods = ("predict", "transform", "predict_proba", "decision_function")
    if (not hasattr(estimator, "fit")) or (
        not any(hasattr(estimator, method) for method in valid_methods)
    ):
        raise ValueError(
            "estimator must be a scikit-learn transformer or "
            "an estimator with the fit and a predict-like (e.g. "
            "predict_proba) or a transform method."
        )

    if get_params and not hasattr(estimator, "get_params"):
        raise ValueError(
            "estimator must be a scikit-learn transformer or an "
            "estimator with the get_params method that allows "
            "cloning."
        )


def _get_inverse_funcs(estimator, terminal=True):
    """Retrieve the inverse functions of an pipeline or an estimator."""
    inverse_func = [False]
    if hasattr(estimator, "steps"):
        # if pipeline, retrieve all steps by nesting
        inverse_func = list()
        for _, est in estimator.steps:
            inverse_func.extend(_get_inverse_funcs(est, terminal=False))
    elif hasattr(estimator, "inverse_transform"):
        # if not pipeline attempt to retrieve inverse function
        inverse_func = [estimator.inverse_transform]

    # If terminal node, check that that the last estimator is a classifier,
    # and remove it from the transformers.
    if terminal:
        last_is_estimator = inverse_func[-1] is False
        all_invertible = False not in inverse_func[:-1]
        if last_is_estimator and all_invertible:
            # keep all inverse transformation and remove last estimation
            inverse_func = inverse_func[:-1]
        else:
            inverse_func = list()

    return inverse_func


def get_coef(estimator, attr="filters_", inverse_transform=False):
    """Retrieve the coefficients of an estimator ending with a Linear Model.

    This is typically useful to retrieve "spatial filters" or "spatial
    patterns" of decoding models :footcite:`HaufeEtAl2014`.

    Parameters
    ----------
    estimator : object | None
        An estimator from scikit-learn.
    attr : str
        The name of the coefficient attribute to retrieve, typically
        ``'filters_'`` (default) or ``'patterns_'``.
    inverse_transform : bool
        If True, returns the coefficients after inverse transforming them with
        the transformer steps of the estimator.

    Returns
    -------
    coef : array
        The coefficients.

    References
    ----------
    .. footbibliography::
    """
    # Get the coefficients of the last estimator in case of nested pipeline
    est = estimator
    while hasattr(est, "steps"):
        est = est.steps[-1][1]

    squeeze_first_dim = False

    # If SlidingEstimator, loop across estimators
    if hasattr(est, "estimators_"):
        coef = list()
        for this_est in est.estimators_:
            coef.append(get_coef(this_est, attr, inverse_transform))
        coef = np.transpose(coef)
        coef = coef[np.newaxis]  # fake a sample dimension
        squeeze_first_dim = True
    elif not hasattr(est, attr):
        raise ValueError(
            "This estimator does not have a %s attribute:\n%s" % (attr, est)
        )
    else:
        coef = getattr(est, attr)

    if coef.ndim == 1:
        coef = coef[np.newaxis]
        squeeze_first_dim = True

    # inverse pattern e.g. to get back physical units
    if inverse_transform:
        if not hasattr(estimator, "steps") and not hasattr(est, "estimators_"):
            raise ValueError(
                "inverse_transform can only be applied onto " "pipeline estimators."
            )
        # The inverse_transform parameter will call this method on any
        # estimator contained in the pipeline, in reverse order.
        for inverse_func in _get_inverse_funcs(estimator)[::-1]:
            coef = inverse_func(coef)

    if squeeze_first_dim:
        coef = coef[0]

    return coef


@verbose
def cross_val_multiscore(
    estimator,
    X,
    y=None,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=None,
    fit_params=None,
    pre_dispatch="2*n_jobs",
):
    """Evaluate a score by cross-validation.

    Parameters
    ----------
    estimator : instance of sklearn.base.BaseEstimator
        The object to use to fit the data.
        Must implement the 'fit' method.
    X : array-like, shape (n_samples, n_dimensional_features,)
        The data to fit. Can be, for example a list, or an array at least 2d.
    y : array-like, shape (n_samples, n_targets,)
        The target variable to try to predict in the case of
        supervised learning.
    groups : array-like, with shape (n_samples,)
        Group labels for the samples used while splitting the dataset into
        train/test set.
    scoring : str, callable | None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        Note that when using an estimator which inherently returns
        multidimensional output - in particular, SlidingEstimator
        or GeneralizingEstimator - you should set the scorer
        there, not here.
    cv : int, cross-validation generator | iterable
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a ``(Stratified)KFold``,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. In all
        other cases, :class:`sklearn.model_selection.KFold` is used.
    %(n_jobs)s
    %(verbose)s
    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int, or str, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

        - None, in which case all the jobs are immediately
          created and spawned. Use this for lightweight and
          fast-running jobs, to avoid delays due to on-demand
          spawning of the jobs
        - An int, giving the exact number of total jobs that are
          spawned
        - A string, giving an expression as a function of n_jobs,
          as in '2*n_jobs'

    Returns
    -------
    scores : array of float, shape (n_splits,) | shape (n_splits, n_scores)
        Array of scores of the estimator for each run of the cross validation.
    """
    # This code is copied from sklearn

    from sklearn.base import clone
    from sklearn.model_selection._split import check_cv
    from sklearn.utils import indexable

    check_scoring = _get_check_scoring()

    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    cv_iter = list(cv.split(X, y, groups))
    scorer = check_scoring(estimator, scoring=scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    # Note: this parallelization is implemented using MNE Parallel
    parallel, p_func, n_jobs = parallel_func(
        _fit_and_score, n_jobs, pre_dispatch=pre_dispatch
    )
    position = hasattr(estimator, "position")
    scores = parallel(
        p_func(
            estimator=clone(estimator),
            X=X,
            y=y,
            scorer=scorer,
            train=train,
            test=test,
            fit_params=fit_params,
            verbose=verbose,
            parameters=dict(position=ii % n_jobs) if position else None,
        )
        for ii, (train, test) in enumerate(cv_iter)
    )
    return np.array(scores)[:, 0, ...]  # flatten over joblib output.


# This verbose is necessary to properly set the verbosity level
# during parallelization
@verbose
def _fit_and_score(
    estimator,
    X,
    y,
    scorer,
    train,
    test,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    error_score="raise",
    *,
    verbose=None,
    position=0
):
    """Fit estimator and compute scores for a given dataset split."""
    #  This code is adapted from sklearn
    from mne.fixes import _check_fit_params
    from sklearn.utils.metaestimators import _safe_split
    from sklearn.utils.validation import _num_samples

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = dt.datetime.now()

    X_train, y_train = _safe_split(estimator, X, y, train)

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_duration = dt.datetime.now() - start_time
        score_duration = dt.timedelta(0)
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            warn(
                "Classifier fit failed. The score on this train-test"
                " partition for these parameters will be set to %f. "
                "Details: \n%r" % (error_score, e)
            )
        else:
            raise ValueError(
                "error_score must be the string 'raise' or a"
                " numeric value. (Hint: if using 'raise', please"
                " make sure that it has been spelled correctly.)"
            )

    else:
        fit_duration = dt.datetime.now() - start_time
        test_score = _score(estimator, X_test, y_test, scorer)
        score_duration = dt.datetime.now() - start_time - fit_duration
        if return_train_score:
            train_score = _score(estimator, X_train, y_train, scorer)

    ret = [train_score, test_score] if return_train_score else [test_score]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_duration.total_seconds(), score_duration.total_seconds()])
    if return_parameters:
        ret.append(parameters)
    return ret

@verbose
def my_cross_val_compo_score(
    estimator,
    X,
    X2,
    y=None,
    y2=None,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=None,
    fit_params=None,
    pre_dispatch="2*n_jobs",
):
    # This code is copied from sklearn

    from sklearn.base import clone
    from sklearn.model_selection._split import check_cv
    from sklearn.utils import indexable

    check_scoring = _get_check_scoring()

    X, y, groups = indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    cv_train = list(cv.split(X, y, groups))

    y = y % 2

    cv_test = list(cv.split(X2, y2, groups))

    # print('cv_train', len(cv_train))
    # print('cv_test', len(cv_test))
    # print('X', X.shape, 'y', y.shape, 'y2', y2.shape)

    try:
        scorer = check_scoring(estimator, scoring=scoring)
    except:
        scorer = scoring

    parallel, p_func, n_jobs = parallel_func(
        _fit_and_compo_score, n_jobs, pre_dispatch=pre_dispatch
    )

    # min_length = min(len(cv_train), len(cv_test))
    # cv_train = cv_train[:min_length]

    scores = parallel(
        p_func(
            estimator=clone(estimator),
            X_train=X[train],
            X_test=X2[test],
            y_train=y[train],
            train=train,
            y_test=y2[test],
            scorer=scorer,
            verbose=verbose,
            fit_params=fit_params,
        )
        for (train, _), (_, test) in zip(cv_train, cv_test)
    )

    return np.array(scores)[:, 0, ...]  # flatten over joblib output.

# This verbose is necessary to properly set the verbosity level
# during parallelization
@verbose
def _fit_and_compo_score(
    estimator,
    X_train,
    X_test,
    y_train,
    train,
    y_test,
    scorer,
    fit_params,
    return_train_score=False,
    verbose=None,
):
    """Fit estimator and compute scores for a given dataset split."""
    from mne.fixes import _check_fit_params

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X_train, fit_params, train)

    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)

        if return_train_score:
            train_score = _score(estimator, X_train, y_train, scorer)

    test_score = _score(estimator, X_test, y_test, scorer)
    ret = [train_score, test_score] if return_train_score else [test_score]

    return ret

@verbose
def my_cross_val_multiscore(
    estimator,
    X,
    X2,
    y=None,
    y2=None,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=None,
    fit_params=None,
    pre_dispatch="2*n_jobs",
):
    # This code is copied from sklearn

    from sklearn.base import clone
    from sklearn.model_selection._split import check_cv
    from sklearn.utils import indexable

    check_scoring = _get_check_scoring()

    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    cv_iter = list(cv.split(X, y, groups))

    y = y % 2
    # y[y==2] = 0
    # y[y==3] = 1

    y2 = y.copy()

    # print('X', X.shape, 'y', y.shape, 'y2', y2.shape)

    try:
        scorer = check_scoring(estimator, scoring=scoring)
    except:
        scorer = scoring

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    # Note: this parallelization is implemented using MNE Parallel
    parallel, p_func, n_jobs = parallel_func(
        _my_fit_and_score, n_jobs, pre_dispatch=pre_dispatch
    )

    position = hasattr(estimator, "position")

    scores = parallel(
        p_func(
            estimator=clone(estimator),
            X=X,
            X2=X2,
            y=y,
            y2=y2,
            scorer=scorer,
            train=train,
            test=test,
            fit_params=fit_params,
            verbose=verbose,
            parameters=dict(position=ii % n_jobs) if position else None,
        )
        for ii, (train, test) in enumerate(cv_iter)
    )

    return np.array(scores)[:, 0, ...]  # flatten over joblib output.

# This verbose is necessary to properly set the verbosity level
# during parallelization
@verbose
def _my_fit_and_score(
    estimator,
    X,
    X2,
    y,
    y2,
    scorer,
    train,
    test,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    error_score="raise",
    *,
    verbose=None,
    position=0
):
    """Fit estimator and compute scores for a given dataset split."""
    #  This code is adapted from sklearn
    from mne.fixes import _check_fit_params
    from sklearn.utils.metaestimators import _safe_split
    from sklearn.utils.validation import _num_samples

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = dt.datetime.now()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X2, y2, test, train)

    # print('X_test', X_test.shape, 'y_test', y_test.shape)

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_duration = dt.datetime.now() - start_time
        score_duration = dt.timedelta(0)
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            warn(
                "Classifier fit failed. The score on this train-test"
                " partition for these parameters will be set to %f. "
                "Details: \n%r" % (error_score, e)
            )
        else:
            raise ValueError(
                "error_score must be the string 'raise' or a"
                " numeric value. (Hint: if using 'raise', please"
                " make sure that it has been spelled correctly.)"
            )

    else:
        fit_duration = dt.datetime.now() - start_time
        test_score = _score(estimator, X_test, y_test, scorer)

        score_duration = dt.datetime.now() - start_time - fit_duration
        if return_train_score:
            train_score = _score(estimator, X_train, y_train, scorer)

    ret = [train_score, test_score] if return_train_score else [test_score]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_duration.total_seconds(), score_duration.total_seconds()])
    if return_parameters:
        ret.append(parameters)
    return ret


def _score(estimator, X_test, y_test, scorer):
    """Compute the score of an estimator on a given test set.

    This code is the same as sklearn.model_selection._validation._score
    but accepts to output arrays instead of floats.
    """
    if y_test is None:
        score = scorer(estimator, X_test)
    # elif np.sum(y_test==2) !=0:
    #     idx_Go = (y_test==0) | (y_test==2)

    #     y = y_test.copy()
    #     y = y % 2
    #     # y[y==2] = 0
    #     # y[y==3] = 1

    #     # print(y[idx_Go].shape, y[~idx_Go].shape)

    #     if np.any(idx_Go):
    #         score_Go = scorer(estimator, X_test[idx_Go], y[idx_Go])
    #     if np.any(~idx_Go):
    #         score_NoGo = scorer(estimator, X_test[~idx_Go], y[~idx_Go])
    #     if ~np.any(idx_Go):
    #         score_Go = np.full_like(score_NoGo, np.nan)
    #     if ~np.any(~idx_Go):
    #         score_NoGo = np.full_like(score_Go, np.nan)

    #     # print('Go', score_Go.shape, 'NoGo', score_NoGo.shape)

    #     score = [score_Go, score_NoGo]
    else:
        score = scorer(estimator, X_test, y_test)


    if hasattr(score, "item"):
        try:
            # e.g. unwrap memmapped scalars
            score = score.item()
        except ValueError:
            # non-scalar?
            pass
    return score
