import sys,os
import numpy as np
from util3 import *

from sklearn.decomposition import PCA
from linear_classification import linear_classifier_learn, report_accuracy

"""
Program to demonstrate feature extraction using PCA ( Principle Component Analysis )

Sample run:
    $ python feature_extraction_pca.py ../dataset/DS3/
"""

def fit_pca(X, n_components):
    """
    A function to perform principle component analysis and returns the learned model
    """
    pca = PCA(n_components = n_components)
    pca.fit(X)
    return pca

def extract_features_pca(pca, X):
    """
    A function to transform the set, given the learned pca model
    """
    _X = pca.transform(X)
    return _X

def main():
    # Reading data
    data_path = sys.argv[1]
    train_X, train_Y, thresh = read_data(data_path + 'train.csv', data_path + 'train_labels.csv')
    test_X, test_Y, _ = read_data(data_path + 'test.csv', data_path + 'test_labels.csv')

    # Fitting pca model and extracting features from train-set and test-set
    model_pca = fit_pca(train_X, 1)
    train_X_ex = extract_features_pca(model_pca, train_X)
    test_X_ex = extract_features_pca(model_pca, test_X)

    train_set = np.hstack((train_X_ex, train_Y))
    test_set = np.hstack((test_X_ex, test_Y))
    data_set = np.vstack((np.hstack((train_X, train_Y)), np.hstack((test_X, test_Y))))

    # Learning a linear classifier
    model = linear_classifier_learn(train_set)

    accuracy, precision, recall, f_score = [-1]*2, [-1]*2, [-1]*2, [-1]*2

    # Class-1 params
    accuracy[0], precision[0], recall[0], f_score[0] =  report_accuracy(test_set, model, thresh, 1.0, 2.0, 1.0)
    # Class-2 params
    accuracy[1], precision[1], recall[1], f_score[1] =  report_accuracy(test_set, model, thresh, 1.0, 2.0, 2.0)

    print "Coeff:", model.coef_
    print "Class-1:", accuracy[0], precision[0], recall[0], f_score[0]
    print "Class-2:", accuracy[1], precision[1], recall[1], f_score[1]

    # Plotting the results
    plot(data_set[:,:-1], extract_features_pca(model_pca, data_set[:,:-1]), data_set[:, -1], model, 'PCA')

if __name__ == '__main__':
    main()
