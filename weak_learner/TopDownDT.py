import numpy as np
import math
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelBinarizer
from diffprivlib.mechanisms import exponential
from .report_noisy_max import report_noisy_argmax

def validate_inputs(X, y):
    # Check that X and y have correct shape
    X, y = check_X_y(X, y, accept_sparse=True)

    # Force binary labels & ignore multiclass problems.
    labelbin = LabelBinarizer()
    y = labelbin.fit_transform(y)[:, 0]

    return X, y, labelbin
    
def _impurity_proxy(a):
    # Gini index

    if a == 0.0 or a == 1.0:
        return 0
    return 4 * a * (1 - a)

class node():
    def __init__(self, tree, start, end, depth):
        self.tree = tree
        self.start = start
        self.end = end
        self.depth = depth
        self.children = []
        self.feature = -1
        self.weight = np.sum(self.tree.sample_weight_[self.start:self.end])

    def split(self, f):
        pos = self.sort_by_feature(f)
        self.feature = f
        self.children = [None, None]
        self.children[0] = node(tree=self.tree, start=self.start, end=pos, depth=self.depth+1)
        self.children[1] = node(tree=self.tree, start=pos, end=self.end, depth=self.depth+1)

    def sort_by_feature(self, f):
        sorted_permutation = np.argsort(self.tree.X_[self.start:self.end, f]) + self.start
        self.tree.X_[self.start:self.end, :] = self.tree.X_[sorted_permutation, :]
        self.tree.y_weighted_[self.start:self.end] = self.tree.y_weighted_[sorted_permutation]
        self.tree.sample_weight_[self.start:self.end] = self.tree.sample_weight_[sorted_permutation]

        return int(self.end - np.sum(self.tree.X_[self.start:self.end, f]))

    def impurity_proxy_improvement(self, f):
        if self.weight == 0:
            return 0
        current_ratio = np.sum(self.tree.y_weighted_[self.start:self.end]) / self.weight
        pos = self.sort_by_feature(f)

        w_0 = np.sum(self.tree.sample_weight_[self.start:pos])
        w_1 = self.weight - w_0
        weight = [w_0, w_1]

        ratio = [0.0, 0.0]
        if weight[0] != 0:
            ratio[0] = np.sum(self.tree.y_weighted_[self.start: pos]) / weight[0]
        if weight[1] != 0:
            ratio[1] = np.sum(self.tree.y_weighted_[pos: self.end]) / weight[1]
        children_weighted_avg = weight[0] * _impurity_proxy(ratio[0]) + weight[1] * _impurity_proxy(ratio[1])
        return self.weight*_impurity_proxy(current_ratio) - children_weighted_avg

    def is_leaf(self):
        if len(self.children) == 0:
            return True
        return False

    def size(self):
        return self.end - self.start

    def label_weight(self, l):
        if l == 1:
            return np.sum(self.tree.y_weighted_[self.start:self.end])
        else:
            return (self.weight - np.sum(self.tree.y_weighted_[self.start:self.end]))

    def label_sub_tree(self):
        if not self.is_leaf():
            self.children[0].label_sub_tree()
            self.children[1].label_sub_tree()
            return
        # privacy budget: 8 * max_splits * eta * zeta with parallel composition
        label_weights = [self.label_weight(0), self.label_weight(1)]
        self.label = report_noisy_argmax(utilities=label_weights, 
                                        epsilon=4*self.tree.max_splits * self.tree.eta_,
                                        sensitivity=2*self.tree.dp_zeta)


class DPTopDownTree(ClassifierMixin, BaseEstimator):
    def __init__(self, dp_epsilon, max_splits=10, max_depth=None, dp_zeta=None,
                global_epsilon=0.95, global_delta=0):

        '''
        Parameters
        ----------

        dp_epsilon : double
            Total privacy budget of the tree.

        dp_zeta : double
            The promised upperbound on total variation distance of the distributions
            over neighbouring datasets. If `None', will be specified in fit().
        '''
        self.global_delta = global_delta
        self.global_epsilon = global_epsilon
        self.dp_epsilon = dp_epsilon
        self.dp_zeta = dp_zeta
        self.max_splits = max_splits
        self.max_depth = max_depth
        if max_depth is None:
            self.max_depth = max_splits


    def fit(self, X, y, sample_weight=None):
        # Check that X and y have correct shape
        # Currently only supports X as an numpy narray
        X, y = check_X_y(X, y, accept_large_sparse=False)

        X, y, self.binarizer_, = validate_inputs(X, y)
        self.classes_ = self.binarizer_.classes_
        self.n_, self.d_ = X.shape

        # Cloning is necessary, as we will reorder X and y
        if sample_weight is None:
            self.sample_weight_ = np.ones(self.n_) / self.n_
        else:
            self.sample_weight_ = np.array(sample_weight)
        self.X_ = np.array(X, dtype=np.byte)
        self.y_weighted_ = y * self.sample_weight_
        
        self.total_number_splits_ = 0
        
        if self.dp_zeta is None:
            self.dp_zeta = 1 / self.n_
        self.eta_ = self.dp_epsilon / (16 * self.max_splits * self.dp_zeta)


        self.root_ = node(tree=self, start=0, end=self.n_, depth=0)

        candidate_list = []
        candidate_utility = []

        for f in range(self.d_):
            candidate_list.append((self.root_, f))
            candidate_utility.append(self.root_.impurity_proxy_improvement(f))
        
        while(self.total_number_splits_ < self.max_splits and len(candidate_list) != 0):
            candidate_index = -1
            # privacy budget: 8 * eta * zeta 
            # will add up to max_splits * 8*eta*zeta by simple composition
            expo_mech = exponential.Exponential(
                epsilon=self.eta_, 
                sensitivity=4*self.dp_zeta, 
                utility=candidate_utility)
            candidate_index = expo_mech.randomise()

            current, f = candidate_list[candidate_index]
            current.split(f)

            pos = int(candidate_index / self.d_) * self.d_
            candidate_list = candidate_list[:pos] + candidate_list[pos+self.d_:]
            candidate_utility = candidate_utility[:pos] + candidate_utility[pos+self.d_:]
            if (current.depth < self.max_depth):
                for i in range(2):
                    for ff in range(self.d_):
                        candidate_list.append((current.children[i], ff))
                        candidate_utility.append(current.children[i].impurity_proxy_improvement(ff))

            self.total_number_splits_ = self.total_number_splits_ + 1

        #label the tree
        self.root_.label_sub_tree()

        # delete unncessary attributes to free up memory
        self.X_ = None
        self.y_weighted_ = None
        self.sample_weight_ = None            

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
        return self.binarizer_.inverse_transform(y_pred)
