from diffprivlib.mechanisms import exponential
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import unique_labels
from .RandomTree import RandomTree

def validate_inputs(X, y):
    # Check that X and y have correct shape
    # Currently only accept numpy ndarrays for X
    X, y = check_X_y(X, y, accept_sparse=False, accept_large_sparse=False)

    # Force binary labels & ignore multiclass problems.
    labelbin = LabelBinarizer()
    y = labelbin.fit_transform(y)[:, 0]

    return X, y, labelbin

class DPRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, dp_epsilon=0.1, max_depth=5, max_random_trees=100, 
                max_splits=10, dp_zeta=None, extra_random=True, labeling_budget_ratio=0.0,
                global_epsilon=0.95, global_delta=0):
        
        '''
        Parameters
        ----------

        epsilon : double
            Total privacy budget of the tree.

        zeta : double
            The promised upperbound on total variation distance of the distributions
            over neighbouring datasets. If `None', will be specified in fit().
        
        extra_random : boolean
            If True we will label random trees leafs randomly and if False we will spend
            `labeling_budget_ratio * epsilon' of the budget on them.

        labeling_budget_ratio : double
            Should be in (0, 1). 
        '''

        self.dp_epsilon = dp_epsilon
        self.dp_zeta = dp_zeta
        self.global_epsilon = global_epsilon
        self.global_delta = global_delta
        self.max_depth = max_depth
        self.max_random_trees = max_random_trees
        self.max_splits = max_splits
        self.extra_random = extra_random
        self.labeling_budget_ratio = labeling_budget_ratio

    def fit(self, X, y, sample_weight=None):
        # Check that X and y have correct shape
        X, y, self.binarizer_, = validate_inputs(X, y)
        
        # Store the classes seen during fit
        self.classes_ = self.binarizer_.classes_

        self.n_, self.d_ = X.shape

        if sample_weight is None:
            sample_weight = np.ones(self.n_) / self.n_

        if self.dp_zeta is None:
            self.dp_zeta = 1 / self.n_

        trees = []
        utility = []

        for _ in range(self.max_random_trees):
            random_tree = RandomTree(max_depth=self.max_depth, 
                                    max_splits=self.max_splits,
                                    extra_random=self.extra_random,
                                    epsilon=(self.dp_epsilon*self.labeling_budget_ratio)/self.max_random_trees,
                                    zeta=self.dp_zeta)

            random_tree.fit(X, y, sample_weight)
            pred_y = random_tree.predict(X)

            trees.append(random_tree)
            utility.append(np.sum(np.array(pred_y == y, dtype=int) * sample_weight))


        expo_mech = exponential.Exponential(epsilon=self.dp_epsilon * (1 - self.labeling_budget_ratio),
                                            sensitivity=2*self.dp_zeta, utility=utility)
        candidate_index = expo_mech.randomise()
        self.tree_ = trees[candidate_index]

        return self

    def predict(self, X):
        check_is_fitted(self, ['tree_'])

        # Input validation
        X = check_array(X)

        return self.binarizer_.inverse_transform(self.tree_.predict(X))

    def _get_tags(self):
        return {'poor_score': True}