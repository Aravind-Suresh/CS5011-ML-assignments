import sys,os
import numpy as np
from util2 import *
import random

"""
Program to perform data imputation given the path to data

Sample run:
    $ python data_imputation.py ../dataset/communities.data ../dataset/CandC
"""

def data_filter(data):
    """
    A utility function to filter the given data, removing non-predictive features
    """
    # First 5 columns are non-predictive
    # Last column is the goal
    _data = data[:, 4:] # np.hstack((data[:,:3], data[:,4:]))
    return _data

def data_impute(arr):
    """
    A utility function to impute the data
    Missing values are replaced with median of the input values for that feature
    """
    arr = np.array(arr)
    city_labels = arr[:,3]
    arr_int_vals = np.hstack((arr[:,:3],arr[:,4:]))
    # x = np.where(arr_int_vals== '?')
    # print np.unique(x[1])
    # Replacing '?' with a sentinel value, -1 in this case
    arr_int_vals[arr_int_vals=='?'] = '-1'
    arr_int_vals_ = np.array(arr_int_vals,dtype=np.float32)
    temp = arr_int_vals_.copy()
    for i in range(temp.shape[1]):
        # Using the median to replace missing values
        temp[:,i][temp[:,i] == -1] = np.median(temp[:,i][temp[:,i] > -1])
        # print np.where(temp[:,i] == -1)
    temp_filled = np.array(temp, dtype=np.str)
    city_labels = city_labels.reshape((-1,1))
    data_final = np.hstack((temp_filled[:,:3],city_labels,temp[:,3:]))

    data_final_filtered = data_filter(data_final)

    return data_final_filtered.tolist()

def generate_splits(data, test_to_train_ratio, num):
    """
    A utility function to split data into 2 sets ( train and test ), and generate 'num' different configs.
    """
    total = len(data)
    test_count = int(test_to_train_ratio*total)
    arr = []
    data = np.array(data)
    for i in range(1, num+1):
        idx = range(total)
        random.shuffle(idx)
        test_idx = idx[:test_count]
        train_idx = idx[test_count:]
        arr.append((data[train_idx], data[test_idx]))

    return arr

def main():
    # Getting inputs
    fin_name = sys.argv[1]
    fout_name_prefix = sys.argv[2]
    test_to_train_ratio = 0.2
    n_splits = 5

    # Reading data
    arr = read_data(fin_name)

    # Performing data imputation
    data_list = data_impute(arr)
    write_data(fout_name_prefix + '.csv', data_list)
    arr = generate_splits(data_list, test_to_train_ratio, n_splits)

    # Writing data
    for i in range(1, n_splits + 1):
        write_data(fout_name_prefix + '-train' + str(i) + '.csv', arr[i-1][0])
        write_data(fout_name_prefix + '-test' + str(i) + '.csv', arr[i-1][1])

    print 'Done'

if __name__ == '__main__':
    main()
