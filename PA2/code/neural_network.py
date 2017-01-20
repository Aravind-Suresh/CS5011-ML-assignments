
"""
Contains the required class definitions for a 3-layered neural network.
"""

# Importing libraries
import sys,os
import numpy as np
import sklearn
import csv
from util import *

class Transformer:

    """
    Transformer is a class to transform and center the inputs before feed-forwarding
    the data.
    It performs Min-Max normalization and squashes inputs to [-1, 1].
    """

    def __init__(self, x):
        """
        Computes the required params for the linear transformation.
        """
        x_min = x.min()
        x_max = x.max()
        m = -0.5*(x_max + x_min)
        s = 2./(x_max - x_min)

        self.params = { 'm': m, 's': s }

    def transform_input(self, x):
        """
        Transforms the given input
        """
        return self.params['s']*(x+self.params['m'])

class Network:

    def __init__(self, config):
        # self.n_layers = len(sizes) # 96, 20, 4

        # Required configuration
        self.config = config
        nn_hdim = self.config.nn_hidden_dim

        # Random initialization of the model
        W1 = np.random.randn(self.config.nn_input_dim, nn_hdim) / np.sqrt(self.config.nn_input_dim)
        # W1 = 2*np.random.random((self.config.nn_input_dim, nn_hdim))-1
        b1 = np.random.random((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, self.config.nn_output_dim) / np.sqrt(nn_hdim)
        # W2 = 2*np.random.random((nn_hdim, self.config.nn_output_dim))
        b2 = np.random.random((1, self.config.nn_output_dim))

        self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2 }

        # print self.model

        # Activation function initialization
        self.A = Collections.sigmoid
        self.Ad = Collections.sigmoid_prime

    def feed_forward(self, x):
        """
        Function to feed-forward the neural network and return the output.
        """
        a = z = x
        A = []
        Z = []
        for W, b in zip(self.weights, self.biases):
            z = a.dot(W) + b
            Z.append(z)
            a = self.A(z)
            A.append(a)
        # Applying softmax function on the output of the last layer
        a = softmax(z)
        A[-1] = a
        return Z, A

    def predict(self, X):
        """
        Function to predict the label given the input.
        """
        W1 = self.model['W1']
        b1 = self.model['b1']
        W2 = self.model['W2']
        b2 = self.model['b2']

        z1 = X.dot(W1) + b1
        a1 = Collections.sigmoid(z1)
        z2 = a1.dot(W2) + b2

        # Applying softmax
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Computing the labels
        labels = np.argmax(probs, axis=1)
        return labels

    def cost(self, X, Y):
        """
        Function returns the cost of prediction, given the predicted and the actual values.
        """
        P = self.predict(X)
        d = P - Y
        cost = 0.5*np.linalg.norm(d)
        return cost

    def accuracy(self, X, Y):
        """
        Utility function to return the accuracy given the input, output.
        """
        # Obtaining the predicted labels
        P = self.predict(X)
        return (100.*np.sum(P == Y)/Y.shape[0])

    def gradient_descent(self, X, Y, n_iter=10000):
        """
        Function to train the neural network using Gradient descent algorithm.
        """

        W1 = self.model['W1']
        b1 = self.model['b1']
        W2 = self.model['W2']
        b2 = self.model['b2']

        # Training for the given no. of iterations ( n_iter )
        for i in range(n_iter):
            z1 = X.dot(W1) + b1
            a1 = Collections.sigmoid(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            n_samples = len(X)

            delta3 = probs
            # Accounting for softmax layer
            delta3[range(n_samples), Y] -= 1
            # Backpropagation
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * a1 * (1-a1)
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.config.reg_lambda * W2
            dW1 += self.config.reg_lambda * W1

            # Gradient descent parameter update
            W1 += -self.config.alpha * dW1
            b1 += -self.config.alpha * db1
            W2 += -self.config.alpha * dW2
            b2 += -self.config.alpha * db2

            self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2 }

            if not i % 3000 and i:
                self.config.alpha *= 0.5
                print 'Alpha changed:', self.config.alpha

            if not i % 100:
                print 'Iter:', i, 'Cost:', self.cost(X, Y)#, 'Acc:', self.accuracy(test_X, test_Y_labels)

class Collections:
    """
    Static class to contain functions.
    """
    @staticmethod
    def sigmoid(x):
        return 1./(1+np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        s = Collections.sigmoid(x)
        return s*(1-s)
