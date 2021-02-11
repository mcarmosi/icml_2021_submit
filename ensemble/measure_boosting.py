"""Measure Boosting

This module contains boosting estimators for classification based on
un-normalized measures, instead of probability distributions.

The module structure is the following:

- The ``BaseMeasureBoosting`` base class implements a common
  ``fit`` method for all the estimators in the module.

- `NonAdaBoost` implements non-adaptive boosting. TODO Cite.

- `BregBoost` implements smooth non-adaptive boosting via interleaved
  Bregman projections to the set of dense measures. TODO Cite.

- `LazyBregBoost` implements smooth non-adaptive boosting via ``lazy``
  Bregman projections to the set of dense measures. TODO Cite.

"""
# This is a BSD 3 Clause piece of software, because we got it by
# hacking apart the sk-learn distributed weightBoosting file.

# Labels MUST be supplied to this class as 0,1, nothing else will
# work!! I'm sure there's a way to fix that "nicely", but I definitely
# don't know it.

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.special import xlogy

from sklearn.ensemble._base import BaseEnsemble

from sklearn.base import ClassifierMixin

from sklearn.tree import DecisionTreeClassifier

from utils import measures as ms

from sklearn.utils import (
    check_array,
    check_random_state,
    check_X_y,
    _safe_indexing, ## we can maybe delete this, it's for regression?
)

from sklearn.utils.extmath import (
    softmax,
    stable_cumsum,
)

from sklearn.metrics import accuracy_score

from sklearn.utils.validation import (
    check_is_fitted,
    # check_sample_weight is bogus, need to replace with custom
    # bounded-measure code.
    _check_sample_weight,
    has_fit_parameter,
    _num_samples,
)

__all__ = [
    'NonAdaBoost',
    'BregBoost',
    'LazyBregBoost'
]


class BaseMeasureBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Base class for MeasureBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.

    """

    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 estimator_params=tuple(),
                 learning_rate=0.1,
                 random_state=None,
                 keep_transcript=False):
        self.learning_rate = learning_rate
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.learning_rate = learning_rate
        self.random_state = random_state
        self.keep_transcript = keep_transcript

    def _validate_data(self, X, y=None):
        # Accept or convert to these sparse matrix formats so we can
        # use _safe_indexing ---> but WHY
        accept_sparse = ['csr', 'csc']
        if y is None:
            ret = check_array(X,
                              accept_sparse=accept_sparse,
                              ensure_2d=False,
                              allow_nd=True,
                              dtype=None)
        else:
            ret = check_X_y(X, y,
                            accept_sparse=accept_sparse,
                            ensure_2d=False,
                            allow_nd=True,
                            dtype=None,
                            y_numeric=False)
        return ret

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=DecisionTreeClassifier(max_depth=1))

        if not has_fit_parameter(self.base_estimator_, "sample_weight"):
            raise ValueError("%s doesn't support sample_weight."
                             % self.base_estimator_.__class__.__name__)


    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
        """
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        # TODO: throw an error if we have labels that aren't 0,1
        X, y = self._validate_data(X, y)

        # re-code scores: 0 -> -1, 1 -> 1
        y = np.where(y == 0, -1, y)
        
        # TODO: check that sample_weight is between 0 and 1 for each
        # sample. we only work with BOUNDED measures.

        sample_m_proj = _check_sample_weight(sample_weight, X, np.float64)
        sample_m_raw = _check_sample_weight(sample_weight, X, np.float64)
        if np.any(sample_m_proj < 0):
            raise ValueError("sample_weight cannot contain negative weights")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.run_score = np.zeros(X.shape[0])

        if self.keep_transcript:
            # transcripts are #rounds x #samples
            self.raw_m_transcript = np.zeros((self.n_estimators, X.shape[0]))
            self.proj_m_transcript = np.zeros((self.n_estimators, X.shape[0]))
            self.score_transcript = np.zeros((self.n_estimators, X.shape[0]))
        
        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # Boosting step
            sample_m_raw, sample_m_proj, score = self._boost(
                iboost,
                X, y,
                sample_m_raw,
                sample_m_proj,
                random_state)

            self.run_score += score
            
            if self.keep_transcript:
                self.raw_m_transcript[iboost,]  = sample_m_raw
                self.proj_m_transcript[iboost,] = sample_m_proj
                self.score_transcript[iboost,] = score
            
        return self

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.

        Warning: This method needs to be overridden by subclasses.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        pass

    def staged_score(self, X, y, sample_weight=None):
        """Return staged scores for X, y.

        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            Labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Yields
        ------
        z : float
        """
        X = self._validate_data(X)

        for y_pred in self.staged_predict(X):
            yield accuracy_score(y, y_pred, sample_weight=sample_weight)

    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        X = self._validate_data(X)

        pred = self.decision_function(X)

        pred = np.where(pred < 0, 0, 1)
        
        return pred

    

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Yields
        ------
        y : generator of array, shape = [n_samples]
            The predicted classes.
        """
        X = self._validate_data(X)

        for pred in self.staged_decision_function(X):
            yield np.where(pred < 0, 0, 1)

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        score : array, shape = [n_samples, k]
            values closer to -1 or 1 mean more like the 0 or 1 respectively
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        vote = sum(estimator.predict(X) for estimator in self.estimators_)
        score = vote / self.n_estimators

        return score

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Yields
        ------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        pred = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_,
                                     self.estimators_):
            norm += weight

            current_pred = estimator.predict(X)

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred
                
            tmp_pred = np.copy(pred)
            yield (tmp_pred / norm)


            
class NonAdaBoost(ClassifierMixin, BaseMeasureBoosting):
    """A NonAdaBoost classifier.

    A NonAdaBoost [TODO**CITE] classifier is a meta-estimator that
    begins by fitting a classifier on the original dataset and then
    fits additional copies of the classifier on the same dataset but
    where the weights of incorrectly classified instances are adjusted
    such that subsequent classifiers focus more on difficult
    cases. The amount of adjustment of these weights in NOT adaptive.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is ``DecisionTreeClassifier(max_depth=1)``.

    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : array of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : array of floats
        Classification error for each estimator in the boosted
        ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances if supported by the ``base_estimator``.
    """
    def __init__(self,
                 base_estimator=None, *,
                 n_estimators=50,
                 learning_rate=1.,
                 random_state=None):
        

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    # We can bring back algorithm selection later if necessary --
    # we'll eventually want to toggle between normal NonAdaBoost and
    # Bregman-projected NonAdaBoost...

    def _boost(self, iboost, X, y, sample_m_raw, sample_m_proj, random_state):
        """Implement a single step of boosting.

        Perform a single boost according to the NonAdaBoost algorithm
        and return the updated sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_m : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The current random number generator

        Returns
        -------
        sample_measure_raw : array-like of shape (n_samples,)
            The reweighted sample weights, no post-processing.

        sample_measure_proj : array-like of shape (n_samples,)
            The reweighted sample weights, with post-processing.

        """
        estimator = self._make_estimator(random_state=random_state)
        
        # Get a distribution /just/ before calling the weak learner
        # TODO: switch for a measures.normalize function later
        sample_dist = (sample_m_raw / np.sum(sample_m_raw))
        
        estimator.fit(X, y, sample_weight=sample_dist)
        
        y_predict = estimator.predict(X)

        # Instances correctly classified
        correct = y_predict == y

        # initialize to sane default
        sample_m_next = np.ones(X.shape[0])

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_m_next = sample_m_raw * np.exp(-self.learning_rate * correct *
                                           (sample_dist > 0))
            
        return sample_m_next, sample_m_next, correct


# Neither sample_measure nor X are instance variables so I think this
# "phrasing" makes sense.

# NonAdaBoost with measures /is/ the same as NonAdaBoost with
# distributions, right...?

class BregBoost(ClassifierMixin, BaseMeasureBoosting):
    """A BregBoost classifier.

    A NonAdaBoost [TODO**CITE] classifier is a meta-estimator that
    begins by fitting a classifier on the original dataset and then
    fits additional copies of the classifier on the same dataset but
    where the weights of incorrectly classified instances are adjusted
    such that subsequent classifiers focus more on difficult
    cases. The amount of adjustment of these weights in NOT adaptive.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is ``DecisionTreeClassifier(max_depth=1)``.

    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : array of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : array of floats
        Classification error for each estimator in the boosted
        ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances if supported by the ``base_estimator``.
    """
    def __init__(self,
                 base_estimator=None, *,
                 n_estimators=50,
                 learning_rate=1.,
                 density=.01,
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)
        
        self.density = density

    def _boost(self, iboost, X, y, sample_m_raw, sample_m_proj, random_state):
        """Implement a single step of boosting.

        Perform a single boost according to the BregBoost algorithm
        and return the updated sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_measure : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The current random number generator

        Returns
        -------
        sample_measure : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The classification error for the current boost.

            If None then boosting has terminated early.

        """
        estimator = self._make_estimator(random_state=random_state)
        
        
        # Get a distribution /just/ before calling the weak learner
        sample_dist = (sample_m_proj / np.sum(sample_m_proj))
        
        estimator.fit(X, y, sample_weight=sample_dist)

        y_predict = estimator.predict(X)

        # Instances incorrectly classified
        correct = y_predict == y

        # Weighted error on requested distribution.
        estimator_error = np.average(correct, weights=sample_dist)

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_dist, 1., 0.

        # Do we even need these?
        estimator_weight = 1

        sample_m_next_raw = np.ones(X.shape[0])
        
        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_m_next_raw = sample_m_proj * np.exp(-self.learning_rate * correct *
                                    (sample_dist > 0))

        # Need to put in the bregman projection right here.

        # TODO: flag for CVX vs sorting?

        # _, sample_m_next_proj = ms.bregman_projection_cvx(sample_m_next_raw,
        #                                         self.density,
        #                                         distribution=False)

        # on the other hand, this works great! Density-sequence of
        # intermediate measures seems to accord with theoretical
        # predictions too!

        print("Density is:" + ms.density(sample_m_next_raw))
        sample_m_next_proj = ms.measures_bregman_projection(sample_m_next_raw,        self.density)
        
        return sample_m_next_raw, sample_m_next_proj, correct

    
class LazyBregBoost(ClassifierMixin, BaseMeasureBoosting):
    """A BregBoost classifier.

    A NonAdaBoost [TODO**CITE] classifier is a meta-estimator that
    begins by fitting a classifier on the original dataset and then
    fits additional copies of the classifier on the same dataset but
    where the weights of incorrectly classified instances are adjusted
    such that subsequent classifiers focus more on difficult
    cases. The amount of adjustment of these weights in NOT adaptive.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is ``DecisionTreeClassifier(max_depth=1)``.

    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : array of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : array of floats
        Classification error for each estimator in the boosted
        ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances if supported by the ``base_estimator``.
    """
    def __init__(self,
                 base_estimator=None, *,
                 n_estimators=50,
                 learning_rate=1.,
                 density=.01,
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)
        
        self.density = density

    def _boost(self, iboost, X, y, sample_m_raw, sample_m_proj, random_state):
        """Implement a single step of boosting.

        Perform a single boost according to the BregBoost algorithm
        and return the updated sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        random_state : RandomState
            The current random number generator

        Returns
        -------
        sample_measure : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The classification error for the current boost.

            If None then boosting has terminated early.

        """
        estimator = self._make_estimator(random_state=random_state)
        
        # Get a distribution /just/ before calling the weak learner
        sample_dist = (sample_m_proj / np.sum(sample_m_proj))
        
        estimator.fit(X, y, sample_weight=sample_dist)

        y_predict = estimator.predict(X)

        # Instances incorrectly classified
        correct = y_predict == y

        score = correct + self.run_score

        sample_m_next_raw = np.ones(X.shape[0]) * self.density
        
        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_m_next_raw = sample_m_next_raw * np.exp(-self.learning_rate * score * (sample_dist > 0))

        sample_m_next_proj = ms.measures_bregman_projection(sample_m_next_raw, self.density)
        
        return sample_m_next_raw, sample_m_next_proj, correct

