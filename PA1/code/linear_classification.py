import sys,os
import numpy as np
from util1 import *
from sklearn import linear_model

"""
Program to demonstrate linear classification i.e classification using linear regression

Sample run:
    $ python linear_classification.py ../dataset/DS1-train.csv ../dataset/DS1-test.csv
"""

# No. of classes
n_classes = 2
# No. of features
p = 20

def linear_classifier_learn(train_set):
    """
    Function to learn the linear classifier given the train-set and returns the model
    """
    # Extracting X
    X = train_set[:,:-1]

    # Extracting labels
    Y = train_set[:,-1]

    # Training a linear regressor
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    return regr

def report_accuracy(test_set, model, thresh = 0.5, output1 = 0.0, output2 = 1.0, label = 1):
    """
    A utility function to report accuracy
    """
    # Extracting X
    X = test_set[:,:-1]

    # Extracting labels
    Y = test_set[:,-1]

    # Predicted labels
    pred = model.predict(X)
    pred[pred <= thresh] = output1
    pred[pred > thresh] = output2

    accuracy = measure_accuracy(pred, Y)
    precision = measure_precision(pred, Y, label)
    recall = measure_recall(pred, Y, label)
    f_score = measure_f_score(precision, recall)

    return accuracy, precision, recall, f_score

def main():
    # Getting inputs
    fin_train, fin_test = sys.argv[1:3]

    # Reading data
    train_set, test_set = read_dataset(fin_train, fin_test, p)
    model = linear_classifier_learn(train_set)

    print model.coef_
    print 'Class-0:'
    accuracy, precision, recall, f_score =  report_accuracy(test_set, model, 0.5, 0.0, 1.0, 0)
    print 'A =', accuracy, 'P =', precision, 'R =', recall, 'F =', f_score
    print 'Class-1:'
    accuracy, precision, recall, f_score =  report_accuracy(test_set, model, 0.5, 0.0, 1.0, 1)
    print 'A =', accuracy, 'P =', precision, 'R =', recall, 'F =', f_score

    write_coeff_to_csv('../report/coeffs/linear_classification_coeff.csv', [0.5] + model.coef_)

if __name__ == '__main__':
    main()
