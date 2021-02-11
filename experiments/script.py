import sys, os
sys.path.append(os.getcwd())

import numpy as np
from sklearn.model_selection import GridSearchCV

from ensemble import measure_boosting as mb

from weak_learner.OneRule import OneRule
from weak_learner.DPRandomForest import DPRandomForest
from weak_learner.TopDownDT import DPTopDownTree

from data.parse_adult import parse_adult
from data.parse_codrna import parse_codrna

from utils.grid_gen import *
from utils.plotting import *

def parse_dataset(dataset):
    parse = globals()["parse_%s"%dataset]
    X, y, _, _ = parse("data/raw/%s/%s.csv"%(dataset,dataset))
    if type(X) != np.ndarray:
        X = X.toarray()

    return X, y

def apply_random_permutation(X, y):
    np.random.seed(0)
    perm = np.random.permutation(X.shape[0])
    X = X[perm,]
    y = y[perm]
    return X, y

def wkl_grid_generator(grid_axis, weak_learner, n_sample, n_features):
    grid = [grid_point(n_sample, n_features, **x) for x in grid_product(grid_axis)]
    if weak_learner == 'OneRule':
        return [add_OneRule(**x) for x in grid]
    elif weak_learner == 'TopDown':
        return [add_DPTopDown(**x) for x in grid]
    else:
        return [add_DPRandomForest(**x) for x in grid]

def store_results(clf_LazyBreg_lrg_DP, dataset, weak_learner):
    poc_df = cv_as_df(clf_LazyBreg_lrg_DP)
    poc_df = prettyprint_privacy(poc_df)
    poc_df.to_csv('results/%s_%s_results.csv'%(dataset, weak_learner))

def run(dataset, weak_learner):
    # NonAdaBoostEstimator = mb.NonAdaBoost()
    LazyBB_Estimator = mb.LazyBregBoost()

    X, y = parse_dataset(dataset)
    X, y = apply_random_permutation(X, y)

    wkl_grid_axis = {'OneRule': dp_lrg_grid, 
                     'TopDown': dp_lrg_TD_grid, 
                     'RandomForest': dp_lrg_RF_grid
                    }

    wkl_grid = wkl_grid_generator(wkl_grid_axis[weak_learner], weak_learner, X.shape[0], X.shape[1])

    clf_LazyBreg_lrg_DP = GridSearchCV(LazyBB_Estimator, wkl_grid, n_jobs=-1, verbose=10)
    clf_LazyBreg_lrg_DP.fit(X, y)

    store_results(clf_LazyBreg_lrg_DP, dataset, weak_learner)

if __name__ == '__main__':
    dataset = sys.argv[-2]          # creditcard, adult, skin, congressional, mushroom, codrna
    weak_learner = sys.argv[-1]     # OneRule, TopDown, RandomForest
    run(dataset, weak_learner)

# Running the experiment
# ex. : python experiments/script.py congressional TopDown

# Running the expermiment with profiling
# ex. : 
# python -m cProfile -o profile.pstats experiments/script.py congressional TopDown
# gprof2dot -f pstats profile.pstats | dot -Tsvg -o mine.svg
# https://stackoverflow.com/questions/582336/how-can-you-profile-a-python-script
