import sys, os
sys.path.append(os.getcwd())

from weak_learner.OneRule import OneRule
import numpy as np
np.random.seed(0)
from sklearn.model_selection import (cross_val_score, cross_validate)

from data.parse_adult import parse_adult
from data.parse_codrna import parse_codrna


def run(dataset):
    parse = globals()["parse_%s"%dataset]
    X, y, _, _ = parse("../data/raw/%s/%s.csv"%(dataset,dataset))
    p = np.random.permutation(X.shape[0])
    X = X[p, ]
    y = y[p]
    clf_np_base = OneRule()
    baseline_cv = cross_val_score(clf_np_base, X.todense(), y, cv=100)
    print('cv_score_avg: ', baseline_cv.mean())
    print('cv_score_std: ', np.std(baseline_cv))
    # print(baseline_cv)


if __name__ == '__main__':
    dataset = sys.argv[-1]
    run(dataset)
