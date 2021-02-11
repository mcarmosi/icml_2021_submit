"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    _check_sample_weight,
)

from diffprivlib.mechanisms import exponential

# scipy.sparse.hstack(blocks, format=None, dtype=None) to replace concatenate

def validate_inputs(X, y):
    # Check that X and y have correct shape
    X, y = check_X_y(X, y, accept_sparse=True)

    # Force binary labels & ignore multiclass problems.
    labelbin = LabelBinarizer()
    y = labelbin.fit_transform(y)[:, 0]

    return X, y, labelbin

def score_features(X, y, verbose, sample_weight = None):
    # Score the candidates according to sample_weight   
    n, d = X.shape

    if verbose:
        print('Input is ', n, ' rows and ', d, ' cols')

    # Validate sample weights
    if sample_weight is None:
        sample_weight = np.ones(n)/n
    else:
        sample_weight = _check_sample_weight(sample_weight, X)
        
    total_weight = sample_weight.sum()

    # Q: is there any risk of floating point error?
    # A: For truly huge n, yes. But n will not get that huge and still
    # fit data into RAM, I think. Ignore this for now.
    
    A = X.transpose().dot(y * sample_weight)
    B = np.array(X.transpose().dot(sample_weight))
    c = (y * sample_weight).sum()
    acc_const_features = np.array([c/total_weight, 1-c/total_weight])
    acc_pos_features = (total_weight - B - c + 2*A) / total_weight
    acc_neg_features = 1 - acc_pos_features
    candidates = np.concatenate((acc_pos_features, acc_neg_features,
                                 acc_const_features), axis=0)
    return candidates, d

def to_signed_feature(candidate, d):
    # literal code and dimension of X to polarity and feature
    sign = 1
    if candidate >= d and candidate < 2*d:
        # I *think* there was an off-by-one error here for
        # negative-polarity features; please double-check.
        candidate = candidate - d
        sign = -1
    elif candidate >= 2*d:
        candidate = -1
        if candidate == 2*d + 1:
            sign = -1
    return sign, candidate

# Assuming X is scipy.sparse.csr.csr_matrix
# if the best feature was a positive feature return sign = 1, candidate = feature index
# if the best feature was a negative feature return sign = -1, candidate = feature index
# if the best feature was a constant feature return sign = ?, candidate = -1

class OneRule(ClassifierMixin, BaseEstimator):
    """ Simplest possible decision stump classifier.

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    def _more_tags(self):
        return {'binary_only': True,
                'requires_y': True}
    
    def fit(self, X, y, sample_weight=None):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """

        X, y, self.binarizer_, = validate_inputs(X, y)

        self.classes_ = self.binarizer_.classes_
        
        candidates, d = score_features(X, y, self.verbose, sample_weight)

        self.n_features_in_ = d
        
        # the difference between baseline and DP:
        literal_choice = candidates.argmax()

        if self.verbose:
            print("Chose literal number:", literal_choice)

        self.sign_, self.feature_ = to_signed_feature(literal_choice, d)

        if self.verbose:
            print("Converted to feature number", self.feature_,
                  " with sign", self.sign_)

        self.err_ = 1 - candidates[literal_choice]
        
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['feature_', 'sign_', 'err_'])

        # Input validation (make sure input is right shape)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            msg = "Number of features %d does not match previous data %d."
            raise ValueError(msg % (X.shape[1], self.n_features_in_))

        if self.verbose:
            print('Predicting with ', self.feature_)
        
        pred = np.ones(X.shape[0])
        if self.feature_ != -1:
            pred = X[:, self.feature_]

        if self.sign_ == -1:
            pred = np.ones(X.shape[0]) - pred

        return self.binarizer_.inverse_transform(pred)


class PrivateOneRule(ClassifierMixin, BaseEstimator):
    """ Simplest possible decision stump classifier.

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, dp_epsilon=1, dp_zeta=1, global_epsilon=0.95,
                 global_delta=0, verbose=False):
        self.dp_epsilon = dp_epsilon
        self.dp_zeta = dp_zeta
        self.global_epsilon = global_epsilon
        self.global_delta = global_delta
        self.verbose = verbose

    def __str__(self):
        output = 'DPOneRule(epsilon = {:.8f}, zeta = {:.8f})' 
        return output.format(self.dp_epsilon, self.dp_zeta)

    def _more_tags(self):
        return {'binary_only': True,
                'requires_y': True}

    def fit(self, X, y, sample_weight=None):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """

        X, y, self.binarizer_, = validate_inputs(X, y)

        self.classes_ = self.binarizer_.classes_

        # Check that X and y have correct shape
        # self.X_, self.y_ = check_X_y(X, y, accept_sparse=True)

        # assume that y is boolean only

        candidates, d = score_features(X, y, self.verbose, sample_weight)

        self.n_features_in_ = d

        # the difference between baseline and DP:
        expo_mech = exponential.Exponential(epsilon=self.dp_epsilon,
                                            sensitivity=self.dp_zeta,
                                            utility=candidates.tolist())
        feature_choice = expo_mech.randomise()

        if self.verbose:
            print("Feature chosen by exp mechanism:", feature_choice)

        self.sign_, self.feature_ = to_signed_feature(feature_choice, d)

        if self.verbose:
            print("Converted to feature number", self.feature_,
                  " with sign", self.sign_)

        self.err_ = 1 - candidates[feature_choice]
        
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['feature_', 'sign_', 'err_'])

        # Input validation (make sure input is right shape)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            msg = "Number of features %d does not match previous data %d."
            raise ValueError(msg % (X.shape[1], self.n_features_in_))

        if self.verbose:
            print('Predicting with ', self.feature_)

        pred = np.ones(X.shape[0])
        if self.feature_ != -1:
            pred = X[:, self.feature_]

        if self.sign_ == -1:
            pred = np.ones(X.shape[0]) - pred

        return self.binarizer_.inverse_transform(pred)
