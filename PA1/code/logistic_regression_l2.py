import sys,os
import numpy as np
from util1 import *

from sklearn import linear_model

"""
Program to demonstrate logistic regression

Sample run:
    $ python logistic_regression_l2.py ../dataset/DS2/data_students/
"""

# Penalty, can be 'l1' or 'l2'
penalty = 'l2'

def logistic_regressor_learn(train_set):
    """
    Function to learn the logistic regressor and return the model learnt
    """
    # Extracting X
    X = train_set[:,:-1]

    # Extracting labels
    Y = train_set[:,-1]

    # Training a logistic regressor
    # Setting C to a very large value => no regularisation
    regr = linear_model.LogisticRegression(penalty = penalty, C = 1e9)
    regr.fit(X, Y)

    return regr

def report_accuracy(test_set, model, labels):
    """
    A utility function to report accuracy
    """
    # Extracting X
    X = test_set[:,:-1]

    # Extracting labels
    Y = test_set[:,-1]

    # Predicted labels
    pred = model.predict(X)
    accuracy, precision, recall, f_score = [], [], [], []

    for label in labels:
        accuracy.append(measure_accuracy(pred, Y))
        p = measure_precision(pred, Y, label)
        precision.append(p)
        r = measure_recall(pred, Y, label)
        recall.append(r)
        f_score.append(measure_f_score(p, r))

    return accuracy, precision, recall, f_score

def read_data(fin_data_name, fin_labels_name):
    """
    A utility function to read the data given the path to input file and labels
    """
    lines1 = open(fin_data_name, 'r').readlines()
    lines2 = open(fin_labels_name, 'r').readlines()
    X, Y = [], []
    lx, ly = -1, -1
    for l in lines1[1:]:
        m = l.split(' ')
        if len(m) == 2:
            lx, ly = int(m[0]), int(m[1])
        else:
            X.append(int(m[0]))
    X = np.array(X).reshape((lx, ly))
    for l in lines2[1:]:
        m = l.split(' ')
        if len(m) == 2:
            lx, ly = int(m[0]), int(m[1])
        else:
            Y.append(int(m[0]))
    Y = np.array(Y).reshape((-1, 1))
    return X, Y

def main():
    # Getting inputs
    data_path = sys.argv[1]
    train_X, train_Y = read_data(data_path + 'Train_features', data_path + 'Train_labels')
    test_X, test_Y = read_data(data_path + 'Test_features', data_path + 'Test_labels')

    train_set = np.hstack((train_X, train_Y))
    test_set = np.hstack((test_X, test_Y))

    # Learning the model
    model = logistic_regressor_learn(train_set)

    # Analysing performance
    accuracy, precision, recall, f_score =  report_accuracy(test_set, model, [-1, 1])

    print "Coeff:", model.coef_
    print "Class-1:", 'A =', accuracy[0], 'P =', precision[0], 'R =', recall[0], 'F =', f_score[0]
    print "Class-2:", 'A =', accuracy[1], 'P =', precision[1], 'R =', recall[1], 'F =', f_score[1]

if __name__ == '__main__':
    main()
