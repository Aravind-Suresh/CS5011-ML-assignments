"""
Program to train a neural network ( without regularization ) on DS2 and report the performance

Sample run:
    $ python neural_network_main.py ../dataset/DS2_train.csv ../dataset/DS2_test.csv
"""

# Importing libraries
import sys,os
import numpy as np
import sklearn
import csv
from util import *
from neural_network import *
from sklearn.metrics import precision_recall_fscore_support

class Config:
    nn_input_dim = 96  # input layer dimensionality
    nn_output_dim = 4  # output layer dimensionality
    nn_hidden_dim = 40 # hidden layer dimensionality
    alpha = 1e-2
    reg_lambda = 0.0 # No regularization

# Extracting train-data, test-data path from commandline
train_data_path = sys.argv[1]
test_data_path = sys.argv[2]

# Reading data
train_data = read_data(train_data_path)
test_data = read_data(test_data_path)
np.random.shuffle(train_data)
n_classes = len(np.unique(train_data[:, -1]))

# Transforming the inputs for training
train_X = np.array(train_data[:, :-1])
transformer = Transformer(train_X)
train_X = transformer.transform_input(train_X)

train_Y_labels = np.uint8(train_data[:, -1])

test_X = test_data[:, :-1]
# Transforming the inputs for testing
test_X = transformer.transform_input(test_X)
test_Y_labels = np.uint8(test_data[:, -1])

# Initializing the network
net = Network(Config)

# Training the network
net.gradient_descent(train_X, train_Y_labels, n_iter=10000)
pred = net.predict(test_X)
accuracy = net.accuracy(test_X, test_Y_labels)
precision, recall, f_score, _ = precision_recall_fscore_support(pred, test_Y_labels, average=None)

# Displaying performance
print 'Best model predictions'
print 'Acc. =', accuracy
print 'Precision =', precision
print 'Recall =', recall
print 'F-score =', f_score
