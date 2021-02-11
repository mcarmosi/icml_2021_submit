import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelBinarizer
from diffprivlib.mechanisms import exponential
from .report_noisy_max import report_noisy_argmax
import secrets

class node():
    def __init__(self, depth, tree):
        self.tree = tree
        self.depth = depth
        self.weights = [0.0, 0.0]
        self.label = -1
        self.children = []
        self.feature = -1

    def split(self, f):
        self.feature = f
        self.children = [None, None]
        self.children[0] = node(tree=self.tree, depth=self.depth+1)
        self.children[1] = node(tree=self.tree, depth=self.depth+1)

    def is_leaf(self):
        if len(self.children) == 0:
            return True
        return False

class RandomTree(ClassifierMixin, BaseEstimator):
    def __init__(self, max_splits=10, max_depth=3, extra_random=True, epsilon=None, zeta=None):
        self.max_splits = max_splits
        self.max_depth = max_depth
        self.extra_random = extra_random
        self.epsilon = epsilon
        self.zeta = zeta
        if extra_random:
            self.epsilon = 0
        else:
            if (zeta is None) or (epsilon is None):
                msg = "privacy budget and sensitivty parameter should be specified."
                raise ValueError(msg)

    def fit(self, X, y, sample_weight):

        self.n_, self.d_ = X.shape
        self.total_number_splits_ = 0
        self.root_ = node(tree=self, depth=0)

        candidate_list = []
        candidate_list.append(self.root_)

        while(self.total_number_splits_ < self.max_splits and len(candidate_list) != 0):
            candidate_index = secrets.randbelow(len(candidate_list))
            current, f = candidate_list[candidate_index], secrets.randbelow(self.d_)
            current.split(f)

            candidate_list[candidate_index] = candidate_list[-1]
            candidate_list.pop()
            
            if (current.depth < self.max_depth):
                candidate_list.append(current.children[0])
                candidate_list.append(current.children[1])

            self.total_number_splits_ = self.total_number_splits_ + 1
        
        if not self.extra_random:
            self.calulate_weights(X, y, sample_weight)
        self.label_sub_tree(self.root_)

        return self


    def calulate_weights(self, X, y, sample_weight):
        for row in range(X.shape[0]):
            x = X[row, ]
            current = self.root_
            while not current.is_leaf():
                current = current.children[int(x[current.feature])]
            current.weights[y[row]] = current.weights[y[row]] + sample_weight[row]

    def label_sub_tree(self, node):
        if not node.is_leaf():             
            self.label_sub_tree(node.children[0])
            self.label_sub_tree(node.children[1])
            return
        if self.extra_random:
            node.label = secrets.randbelow(2)
        else:
            node.label = report_noisy_argmax(utilities=node.weights, 
                                            epsilon=self.epsilon,
                                            sensitivity=2*self.zeta)

    def _single_row_predict(self, x):
        current = self.root_
        while not current.is_leaf():
            current = current.children[int(x[current.feature])]
        return current.label

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['root_'])

        # Input validation (make sure input is right shape)
        X = check_array(X)
        
        if X.shape[1] != self.d_:
            msg = "Number of features %d does not match previous data %d."
            raise ValueError(msg % (X.shape[1], self.d_))

        n = X.shape[0]
        y_pred = np.array([self._single_row_predict(X[i,:]) for i in range(n)])
        return y_pred