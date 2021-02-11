from sklearn.datasets import load_svmlight_file
import pandas
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from data.parse_generic import col_transforming_read
from data.parse_generic import split_train_test
import numpy as np


# pdays is special can be -1
def parse_codrna(fp, verbose=False):
    numeric_cols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    nominal_cols = []
    all_cols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y']
    label_col = ['Y']
    return col_transforming_read(fp, nominal_cols, numeric_cols, label_col, all_cols, 10, ',', 
                                    header=0, verbose=verbose)


if __name__ == "__main__":
    # X, y = load_svmlight_file('raw/cod-rna/cod-rna')
    # all = np.column_stack((X.todense(), y))
    # pandas.DataFrame(all).to_csv("raw/cod-rna/file.csv")

    codrna_X, codrna_y, codrna_fn, codrna_CT = parse_codrna("raw/cod-rna/cod-rna.csv")
    (train_data, train_labels, test_data, test_labels) = split_train_test(codrna_X.todense(), codrna_y, test_ratio=0.1)

    # Save the data for future
    # np.savez('cod-rna.npz', train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)

    # Load the data in another script
    # datafile = np.load('cod-rna.npz')
    # train_data = datafile['train_data']
    # train_labels = datafile['train_labels']
    # test_data = datafile['test_data']
    # test_labels = datafile['test_labels']