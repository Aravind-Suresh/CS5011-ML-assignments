import sys,os
import numpy as np
from util3 import *

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from linear_classification import linear_classifier_learn, report_accuracy

"""
Program to demonstrate feature extraction using LDA ( Linear Discriminant Analysis )

Sample run:
    $ python feature_extraction_lda.py ../dataset/DS3/
"""

def fit_lda(X, Y, n_components):
    """
    A function to perform linear discriminant analysis and returns the learned model
    """
    lda = LinearDiscriminantAnalysis(n_components = n_components)
    lda.fit(X, Y)
    return lda

def extract_features_lda(lda, X):
    """
    A function to transform the set, given the learned lda model
    """
    _X = lda.transform(X)
    return _X

def main():
    # Reading data
    data_path = sys.argv[1]
    train_X, train_Y, thresh = read_data(data_path + 'train.csv', data_path + 'train_labels.csv')
    test_X, test_Y, _ = read_data(data_path + 'test.csv', data_path + 'test_labels.csv')

    # Fitting lda and extracting features from train-set and test-set
    model_lda = fit_lda(train_X, train_Y.ravel(), 1)
    train_X_ex = extract_features_lda(model_lda, train_X)
    test_X_ex = extract_features_lda(model_lda, test_X)

    train_set = np.hstack((train_X, train_Y))
    test_set = np.hstack((test_X, test_Y))
    data_set = np.vstack((train_set, test_set))

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
    plot(data_set[:,:-1], extract_features_lda(model_lda, data_set[:,:-1]), data_set[:, -1], model, 'LDA')

if __name__ == '__main__':
    main()
