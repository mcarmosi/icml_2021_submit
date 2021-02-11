import pandas as pd
import numpy as np
from data.parse_generic import col_transforming_read
from data.parse_generic import split_train_test


def parse_adult(fp, verbose=False):
    # Enumerate all the columns we care about
    nominal_cols = ['workclass', 'education', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex',
                    'native-country']

    numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                    'capital-loss', 'hours-per-week']

    label_col = ['label']

    all_cols = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain',
                'capital-loss', 'hours-per-week',
                'native-country','label']

    return col_transforming_read(fp, nominal_cols, numeric_cols, label_col,
                                 all_cols, 10, ', ', verbose=verbose)

def parse_adult_less(fp):
    # Enumerate all the columns we care about
    nominal_cols = ['workclass', 'education', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex',
                    'native-country']

    numeric_cols = []

    label_col = ['label']

    all_cols = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain',
                'capital-loss', 'hours-per-week',
                'native-country','label']

    return col_transforming_read(fp, nominal_cols, numeric_cols, label_col,
                                 all_cols, 10, ', ')



# if __name__ == '__main__':
#     adults_X, adults_y, adults_fn, adults_CT = parse_adult("raw/adult/adult.data")
#     (train_data, train_labels, test_data, test_labels) = split_train_test(adults_X.todense(), adults_y, test_ratio=0.1)

    # Save the data for future
    # np.savez('adult.npz', train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)

    # Load the data in another script
    # datafile = np.load('adult.npz')
    # train_data = datafile['train_data']
    # train_labels = datafile['train_labels']
    # test_data = datafile['test_data']
    # test_labels = datafile['test_labels']

# HOW TO RUN ON ADULTS (interactive & plot)

# %load grid-search-test.py
# cd data
# %load parse_adult.py
# %load utils.py

# adults_X, adults_y, adults_fn, adults_CT = parse_adult("raw/adult/adult.data") 
# clf.fit(adults_X, adults_y)

# import matplotlib.pyplot as plt 
# plt.ion()
# plot_grid_search(clf.cv_results_, param_grid['n_estimators'],
#                  param_grid['learning_rate'], 'N Estimators', 'Learn Rate') 
