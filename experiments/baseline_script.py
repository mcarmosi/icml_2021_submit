import sys, os
sys.path.append(os.getcwd())
import numpy as np
np.random.seed(0)
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
)

from ensemble import measure_boosting as mb

from weak_learner.OneRule import OneRule

from data.parse_adult import parse_adult
from data.parse_codrna import parse_codrna

from utils.plotting import *


def run(dataset):
    NonAdaBoostEstimator = mb.NonAdaBoost()
    base_learner = OneRule()
    NonAda_OneR_grid = {'n_estimators': [5, 10, 15, 20, 50, 100],
            'learning_rate': [0.1, 0.25, 0.5, 0.75],
            'base_estimator': [base_learner]
            }
    clf_NonAda_full = GridSearchCV(NonAdaBoostEstimator, NonAda_OneR_grid, n_jobs=-1, verbose=10)
    parse = globals()["parse_%s"%dataset]
    X, y, _, _ = parse("../data/raw/%s/%s.csv"%(dataset,dataset))
    if type(X) != np.ndarray:
        X = X.toarray()
    p = np.random.permutation(X.shape[0])
    X = X[p, ]
    y = y[p]
    clf_NonAda_full.fit(X, y)

    poc_df = cv_as_df(clf_NonAda_full)
    poc_df.to_csv('../results/baseline_%s_results.csv'%dataset)


if __name__ == '__main__':
    dataset = sys.argv[-1]
    run(dataset)