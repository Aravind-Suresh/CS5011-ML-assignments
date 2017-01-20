import sys,os
import numpy as np
from util1 import *
from sklearn.neighbors import KNeighborsClassifier

"""
Program to demonstrate a k-NN ( k Nearest neighbors ) classfier

Sample run:
    $ python knn_classifier.py ../dataset/DS1-train.csv ../dataset/DS1-test.csv
"""

# No. of classes
n_classes = 2
# No. of features
p = 20

def knn_classifier_learn(train_set, k = 3):
    """
    Function to learn the k-NN classifier given the train_set and returns the learned model
    """
    # Extracting X
    X = train_set[:,:-1]

    # Extracting labels
    Y = train_set[:,-1]

    # Training K-NN classifier
    neigh = KNeighborsClassifier(n_neighbors = k)
    neigh.fit(X, Y)
    return neigh

def report_accuracy(test_set, model, thresh = 0.5, output1 = 0.0, output2 = 1.0, label = 1):
    """
    A utility function for reporting accuracy
    """
    # Extracting X
    X = test_set[:,:-1]

    # Extracting labels
    Y = test_set[:,-1]

    # Predicted labels
    pred = model.predict(X)

    accuracy = measure_accuracy(pred, Y)
    precision = measure_precision(pred, Y, label)
    recall = measure_recall(pred, Y, label)
    f_score = measure_f_score(precision, recall)

    return accuracy, precision, recall, f_score

def main():
    fin_train, fin_test = sys.argv[1:3]
    train_set, test_set = read_dataset(fin_train, fin_test, p)
    # Training for different values of k = 3..10
    for k in range(3, 11):
        model = knn_classifier_learn(train_set, k)
        accuracy, precision, recall, f_score =  report_accuracy(test_set, model, 0.5, 0.0, 1.0, 0)
        print 'k=', k
        print 'Class-0:', 'accuracy=', accuracy, 'precision=', precision, 'recall=', recall, 'f_score=', f_score
        accuracy, precision, recall, f_score =  report_accuracy(test_set, model, 0.5, 0.0, 1.0, 1)
        print 'Class-1:', 'accuracy=', accuracy, 'precision=', precision, 'recall=', recall, 'f_score=', f_score

if __name__ == '__main__':
    main()
