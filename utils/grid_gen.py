import itertools
import math
import numpy as np

from weak_learner.TopDownDT import DPTopDownTree
from weak_learner.DPRandomForest import DPRandomForest
from weak_learner.OneRule import PrivateOneRule

deltas = [0, 'sLIN']

sml_grid = {'n_estimators': [5, 10, 50, 200],
            'learning_rate': [0.1]
            }
dp_sml_grid = dict(sml_grid,
                   density = [0.1],
                   epsilon = [0.95],
                   delta = deltas)

med_grid = {'n_estimators': [1, 5, 10, 50, 100, 500],
            'learning_rate': [0.01, 0.1, 0.2]
            }
dp_med_grid  = dict(med_grid,
                    density = [0.1, 0.25],
                    epsilon = [0.5, 0.95, 2],
                    delta = deltas)

lrg_grid = {'n_estimators': [5, 10, 20, 50, 100],
            'learning_rate': [0.1, 0.25, 0.5, 0.75],
            }
dp_lrg_grid = dict(lrg_grid,
                   density = [0.1, 0.2, 0.25],
                   epsilon = [2, 1, 0.75, 0.5, 0.25, 0.1],
                   delta = deltas)

tree_grid = {'n_estimators': [5, 10, 20],               # removed 15
            'learning_rate': [0.5, 0.75],
            }
dp_tree_grid = dict(tree_grid,
                   density = [0.2],
                   epsilon = [2, 1, 0.75, 0.5, 0.25, 0.1],
                   delta = deltas)

dp_lrg_TD_grid = dict(dp_tree_grid,
                      max_splits = [3, 10])

dp_lrg_RF_grid = dict(dp_tree_grid,
                      max_splits = [30],                     # added constant max_splits
                      max_depth = [2, 'min(k/2, log(n)-1)'], # removed '(min(k/2, log(n)-1)+2)/2'
                      labeling_budget_ratio = [0.0, 0.5],
                      max_random_trees = [5, 10]) # removed 'log(n)'

def grid_product(init_grid):
    # Take the Cartesian product of a grid, for those parameters that
    # should be combined without restriction. This prepares the grid
    # for further modification, for dependant parameters...
    params = []
     # Always sort the keys of a dictionary, for reproducibility
    items = sorted(init_grid.items())
    keys, values = zip(*items)
    for v in itertools.product(*values):
        params.append(dict(zip(keys,v)))
    return params

def grid_point(n_samples, n_features, density=0.01, epsilon=0.5, delta=0,
                     learning_rate=0.1, n_estimators=100, max_splits=None,
                     labeling_budget_ratio=None, max_random_trees=None, max_depth=None):
    dp_epsilon = drv_compose_budget_search(n_samples, epsilon, delta, n_estimators)
    dp_zeta = 1 / (n_samples * density)
    point = {'density': [density],
            'learning_rate': [learning_rate],
            'n_estimators': [n_estimators],
            'dp_epsilon': dp_epsilon,
            'dp_zeta': dp_zeta,
            'global_epsilon': epsilon,
            'global_delta': delta
            }
    
    if max_splits is not None:          # TopDown or RandomForest
        point['max_splits'] = max_splits
    
    if max_random_trees is not None:    # RandomForest
        point['max_splits'] = max_splits        
        point['max_depth'] = max_depth
        point['max_random_trees'] = max_random_trees
        point['labeling_budget_ratio'] = labeling_budget_ratio

        if max_random_trees == 'log(n)':
            point['max_random_trees'] = int(np.log2(n_samples))
            
        if max_depth == '(min(k/2, log(n)-1)+2)/2':
            point['max_depth']= int(int( min(n_features/2, np.log2(n_samples)-1))/2)

        if max_depth == 'min(k/2, log(n)-1)':
            point['max_depth']= int(min(n_features/2, np.log2(n_samples)-1))

    return point

def drv_compose_budget(n_samples, epsilon, delta, n_rounds):
    if delta == 0:                      # simple composition
        return epsilon / n_rounds
    elif delta == 'sLIN':               # advanced composition, weak delta'
        delta_prime = math.pow(n_samples, -1.1)
    else:                               # advanced compisition, tiny constant delta'
        delta_prime = math.pow(2, -30)
    
    logfactor = math.log( (1 / delta_prime) )
    return epsilon / math.sqrt(2 * n_rounds * logfactor)

def drv_compose_budget_search(n_samples, epsilon, delta, n_rounds):
    def adv_comp_RHS(epsilon_b, n_rounds, logfactor): 
        return np.sqrt(2*n_rounds*logfactor)*epsilon_b \
                    + n_rounds*epsilon_b*(np.exp(epsilon_b)-1) 

    def binary_search_epsilon_b(epsilon, n_rounds, logfactor):
        start = 0.0
        end = epsilon
        for _ in range(100): # large enough
            mid = (start + end) / 2.0
            if adv_comp_RHS(epsilon_b=mid, n_rounds=n_rounds, logfactor=logfactor) > epsilon:
                end = mid
            else:
                start = mid
        return start
    
    if delta == 0:                      # simple composition
        return epsilon / n_rounds
    elif delta == 'sLIN':               # advanced composition, weak delta'
        delta_prime = math.pow(n_samples, -1.1)
    else:                               # advanced compisition, tiny constant delta'
        delta_prime = math.pow(2, -30)
    
    logfactor = math.log( (1 / delta_prime) )

    return binary_search_epsilon_b(epsilon=epsilon, n_rounds=n_rounds, logfactor=logfactor)

def add_OneRule(density, learning_rate, n_estimators,
                dp_epsilon, dp_zeta, global_epsilon, global_delta):
    noised_learner = PrivateOneRule(dp_epsilon=dp_epsilon, dp_zeta=dp_zeta,
                                    global_epsilon=global_epsilon,
                                    global_delta=global_delta)
    return {'density': density,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'base_estimator': [noised_learner]
            }

def add_DPRandomForest(density, learning_rate, n_estimators,
                dp_epsilon, dp_zeta, global_epsilon, global_delta,
                max_depth, max_random_trees, labeling_budget_ratio,
                max_splits):

    extra_random = (labeling_budget_ratio == 0.0)    # check if extra random
    noised_learner = DPRandomForest(dp_epsilon=dp_epsilon, dp_zeta=dp_zeta,
                                    global_epsilon=global_epsilon,
                                    global_delta=global_delta,
                                    max_splits=max_splits, max_random_trees=max_random_trees,
                                    labeling_budget_ratio=labeling_budget_ratio, 
                                    extra_random=extra_random, max_depth=max_depth)
    return {'density': density,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'base_estimator': [noised_learner]
            }

def add_DPTopDown(density, learning_rate, n_estimators,
                dp_epsilon, dp_zeta, global_epsilon, global_delta,
                max_splits):
    noised_learner = DPTopDownTree(dp_epsilon=dp_epsilon, dp_zeta=dp_zeta,
                                    global_epsilon=global_epsilon,
                                    global_delta=global_delta,
                                    max_splits=max_splits)
    return {'density': density,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'base_estimator': [noised_learner]
            }


# For example:
# small_size_grid = [grid_point(3000, **x) for x in grid_product(dp_sml_grid)] 
# [add_OneRule(**x) for x in small_size_grid]
