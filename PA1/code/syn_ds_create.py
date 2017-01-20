import sys,os
import numpy as np
import random
from util1 import *

"""
Program to generate synthetic data of 2 classes, 20 features each, assuming Multivariate Gaussian distributions

Sample run:
    $ python syn_ds_create.py ../dataset/DS1
"""

# No. of classes
n_classes = 2
# No. of features
p = 20
# No. of samples
n_samples = 2000

def generate_covariance_matrix(n):
    """
    A utility function to generate covariance matrix.
    It ensures the matrix to be positive definite and non-spherical.
    """
    A = np.random.random((n, n))
    A = 0.5*(A + np.transpose(A))
    return (A + n*np.eye(n))

def generate_data(p, n_samples, test_to_train_ratio = 0.3):
    """
    A utility function to generate data synthetically.
    Generates data of 2 classes, with Multivariate gaussian distribution.
    Splits it into test-set and train-set, given the test_to_train_ratio.
    """
    m = np.random.random(p)
    d = np.random.random(p)
    _d = d/np.sqrt(np.linalg.norm(d))
    r = 5

    # Centroids of the distributions
    m1 = m
    m2 = m + r*_d

    # Random symmetric positive definite matrix
    cov = generate_covariance_matrix(p)

    # Generating dataset
    x = np.random.multivariate_normal(m1, cov, n_samples)
    y = np.random.multivariate_normal(m2, cov, n_samples)

    label_x = np.ones(n_samples)
    label_y = np.zeros(n_samples)

    X = np.hstack((x, label_x.reshape((-1, 1))))
    Y = np.hstack((y, label_y.reshape((-1, 1))))

    idx = range(n_samples)
    random.shuffle(idx)
    l = int(n_samples*test_to_train_ratio)
    test_idx = idx[:l]
    train_idx = idx[l:]

    train_set = np.vstack((X[train_idx], Y[train_idx]))
    test_set = np.vstack((X[test_idx], Y[test_idx]))

    return train_set, test_set

def main():
    # Generating data
    train_set, test_set = generate_data(p, n_samples)
    dataset_prefix = sys.argv[1]

    # Writing datasets
    write_dataset(dataset_prefix + '-train.csv', train_set)
    write_dataset(dataset_prefix + '-test.csv', test_set)

if __name__ == '__main__':
    main()
